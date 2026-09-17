# Long-Lived Characters, Local Inference: Incremental Memory Maintenance for Game NPCs

Zimu Xu

University of Bern

zimu.xu@unibe.ch

## Abstract

A game character should not have to reread its entire life before every conversation. For locally deployed language-model characters, however, revising a few memories can invalidate a long reusable prefix. The resulting preparation cost competes with both foreground dialogue and the maintenance of other characters. This matters especially when dialogue feeds game-defined actions and value judgments: a fluent but incorrect account of who owns an item, or whether a transfer has already happened, can corrupt the input to otherwise deterministic rules. We study incremen tal memory maintenance for long-lived game NPCs in a quantized Qwen hybrid recurrent– attention model. Our runtime removes superseded attention KV entries, computes replacement records at the true sequence tail, and preserves the continuing recurrent state and unchanged KV. Existing local experiments combine multi-update dialogue replays, fixed-input placement ablations, and attention diagnostics. Independent block composition weakens query-conditioned memory selection without a uniform chunk-initial attention collapse. Truetail updates preserve important current-state and historical bindings across eight scripted maintenance rounds; a placement case recovers the full-refill quantity in three reconstructions, while slot-preserving alternatives repeat a double-subtraction error. Attentiondistribution proximity alone does not explain these semantic diferences. The results moti vate treating a character’s inference state as a maintained, history-dependent resource, rather than only a disposable encoding of its latest memory text.

## 1 Introduction

An NPC that can produce a convincing sentence is not yet an NPC on which a game can depend. Players make plans because they expect resources, relationships, and consequences to retain meaning beyond the current exchange. A promised gift is not an executed transfer. A remembered act of kindness is not evidence that a former ally remains safe. A herb already given away must not be subtracted from the current inventory a second time. These distinctions become mechanically important when open-ended dialogue connects to the systems that govern a game.

Our design premise is that gameplay should be governed by controllable rules, not improvised anew by model generation. Language models interpret flexible requests and character-specific circumstances. An action compiler translates relevant intentions into game-supported activities, while local value and execution rules determine their mechanical consequences. Subjective preferences may difer between characters, and uncertainty may be deliberately introduced through mechanisms such as dice. Neither requires abandoning a coherent economic scale or letting an eloquent sentence create an item. Section 2 explains this premise without assuming that the complete envisioned game has already been implemented or evaluated.

The resulting character needs more than a static persona. It must revise beliefs about people, objects, and places while retaining relevant experience. Memory can be compressed or merged; a seller’s claim can be contradicted by a later test; ownership can change without changing an entity’s identity. Generative-agent research has established the usefulness of experience, reflection, and retrieval for believable behavior (Park et al., 2023). Conversational memory systems likewise manage what to retain and bring into active context (Zhong et al., 2024; Packer et al., 2023). Here we focus on a complementary deployment problem: how can an evolving character context remain computationally available on a player’s machine?

In one measured local configuration, preparing the full dialogue pipeline from cold took 57.5 seconds. This is multi-prefix preparation, not the generation time of one reply. An individual wait may be hidden by prefetching. If dozens of characters need refreshed contexts, however, asynchronous scheduling moves the work without removing it. On a shared local accelerator, background rebuilding competes with the conversation the player is actually having. A memory system that is inexpensive to edit as text can therefore remain expensive to use as a model state. Figure 1 connects this preparation budget to the persistent consequences that make dialogue useful to a game.

![](images/b8fd69b0db4c9f0917784247c067c331309cd875ae515b5091b1d2ee905a3dec.jpg)  
Figure 1: Rule-governed play makes memory maintenance consequential. Open-ended dialogue expresses intentions; compilation connects them to game-supported activities, while game rules govern authorization and consequences. Experiences revise character records. On a shared local device, repeatedly rebuilding long contexts competes with foreground dialogue. The target is to encode revisions while reusing surviving state. This conceptua illustration motivates the workload; it is not a population-throughput measurement or a claim that the complete envisioned game has been implemented.

Prefix caching does not, by itself, solve this problem. Systems such as SGLang reuse shared prefixes across generation calls (Zheng et al., 2024), but updating an early record invalidates the exact reusable prefix from that point onward. Modular reuse goes beyond this restriction (Gim et al., 2024; Yao et al., 2025); the challenge becomes which contextual computation to preserve or repair. Keeping every version as appended text avoids replacement but grows the visible history and preserves conflicting versions. Independently computing blocks avoids some rebuilding, yet breaks the contextual computation that originally related them. Hybrid models add another complication: token-indexed attention KV coexist with recurrent and short-convolution states that cannot be deleted one record at a time.

We investigate an alternative: retain the continuing recurrent state, remove the direct KV of superseded records, and compute replacements at the true tail. The logical memory is editable even though its computational history keeps moving forward. This is intentionally not an exact reconstruction of the latest text. We ask whether it can preserve the semantic continuity needed by a game character while avoiding recomputation of unchanged records.

Our contributions are threefold:

• We formulate a persistent-memory workload for fully local game characters, connecting memory revision to rule-governed dialogue, resource accounting, and action binding.

• We implement incremental hybrid-state maintenance that separates the lifetime of explicit KV records from the continuing recurrent state, without changing model weights.

• We present a case-study and diagnostic evaluation of independent composition, true-tail maintenance, and position-preserving alternatives. The results expose both selectivememory addressing failures and state–event binding errors that aggregate attention similarity does not reliably predict.

The evidence in this study is deliberately organized by experimental cohort, rather than presented as a single benchmark leaderboard. It includes one synthetic character under multiple histories and controlled module replays; multicharacter population evaluation remains future work.

## 2 Design Goal: A Playable Dialogue Interface

## 2.1 Game-defined value before stochasticity

Open-ended language is useful because the player can express intentions beyond a fixed dialogue menu. Its value to a game depends on what those intentions can do. Trading, bargaining, deception, and relationship-building become strategic when the player can learn which conditions matter and trust that investments have persistent meaning.

Our intended value system connects costs and benefits through game-defined exchange relationships: resources, opportunities, and characterspecific preferences must be interpretable on a coherent scale. This does not imply identical prices or perfectly rational NPCs. Trust, need, or attachment may change willingness to pay. The distinction is between a preference with maintained reasons and an arbitrary change of economic scale caused by a new generation. Fairness means that the rules governing such diferences are suficiently stable for the player to form a strategy.

Likewise, controllability does not imply a deterministic world. A dice check has a trigger, a probability model, and defined consequences. It adds uncertainty within a mechanism. Uncontrolled model variability should not silently replace that mechanism. The language model may portray hesitation, pride, or dishonesty; the game must still distinguish what was said, what was intended, what was authorized, and what actually happened.

## 2.2 The harness connects meaning to mechanics

At a high level, the character harness separates three responsibilities. Interpretation resolves the active matter and its context. Compilation describes the character’s potential game-supported activities and, where necessary, the entities and quantities involved. Adjudication applies the game’s value, feasibility, and execution rules. Dialogue communicates the resulting response or plan. The language used in a promise is not itself an execution receipt.

This separation supports a specific product direction: the player can influence a character through flexible language, but cannot simply narrate away resource costs. Proposed mechanics such as disguise-based conversations or costly memory intervention illustrate why the interface is more than a chat window: open language enables unscripted tactics, while world rules determine their boundaries. These mechanics motivate the architecture; their entertainment value and full implementation are not claims of the present experiments.

## 2.3 Memory continuity is a mechanical dependency

Even a deterministic rule receives the wrong input if the character confuses the owner of an item, an intended transfer with a completed one, or a former ally with a current threat. We therefore study memory as a dependency of game-defined decision-making, not merely as a source of colorful anecdotes.

Consider three statements: “I collected three herbs,” “I already gave one away,” and “I currently hold two.” They describe one consistent history, not three independent quantities to combine freely. A small binding error can change a compiled transaction while leaving the prose entirely plausible. This is the kind of error the evaluation prioritizes. Multiple natural replies and reasonable guesses remain acceptable; factual relationships that determine mechanical inputs do not become interchangeable.

## 3 Problem: Mutable Memory on a Shared Local Device

## 3.1 Records, revisions, and character sessions

A character’s memory store contains experience records and entity-cognition records. Experiences include a time description, summary, and retained details. Cognition records identify people, items, and places and describe the character’s current understanding of them. The same entity ID can occur in several memories; revising its description does not create a new person or item.

Let $\mathcal { M } _ { t }$ be the live record set after maintenance step t. An update supplies records $U _ { t }$ and identifies superseded records $D _ { t } \mathbf { \mathrm { : } }$

$$
\mathcal { M } _ { t + 1 } = ( \mathcal { M } _ { t } \setminus D _ { t } ) \cup U _ { t } .\tag{1}
$$

This supports additions, replacement, and manyto-one compression. A revision may remove descriptive detail while preserving event participants and current consequences. The experiments use scripted updates so that the intended facts and lost details are auditable. They do not evaluate an autonomous memory editor.

Three orders need not coincide: when an event happened, when its description was last revised, and where its current KV representation lies. An old experience can receive a recent revision; a recently relevant experience can remain in old, unchanged KV. Successful maintenance must not confuse these orders.

## 3.2 The character-readiness budget

We distinguish three costs. Cold preparation builds reusable contexts and warms the modules. Maintenance revises these contexts after new experience. Hot interaction processes the current sufix and produces a visible reply. Reducing the third does not remove the first two.

Fully local deployment makes their competition concrete. If N characters each require a standalone rebuild costing roughly c device-seconds, $N c$ is a useful first-order accounting of serial work, not a prediction of parallel wall-clock latency. Scheduling, shared prefixes, and batching can change realized costs. They cannot make repeated computation free. A world with many changing characters needs to reduce work as well as hide it. We treat this as deployment motivation; no multi-NPC throughput measurement is reported here.

## 3.3 Semantic continuity, not output identity

For a query $q _ { t } ,$ , a maintained state should support outputs consistent with the relevant live facts and <sup>legitimate</sup> <sup>uncertainty</sup> <sup>in</sup> Mt<sup>.</sup> <sup>We</sup> <sup>do</sup> <sup>not</sup> <sup>require</sup> the same wording, mood, or speculative explanation as a fresh refill. We distinguish:

• State and event fidelity: quantities, ownership, participants, temporal relations, and completed versus proposed actions.

• Contextual usability: whether recent revisions and relevant unchanged memories can inform dialogue and compilation.

• Character expression: coherent, intelligible dialogue, allowing more than one reasonable interpretation or strategy.

Fresh refill is a computational reference, not an infallible narrative oracle. It can make mistakes too.

This emphasis connects to long-term conversational evaluation: LoCoMo studies temporally structured experience, while LongMemEval explicitly tests knowledge updates and temporal reasoning (Maharana et al., 2024; Wu et al., 2025). RULER further distinguishes basic retrieval from tracing and aggregation in long contexts (Hsieh et al., 2024). These works motivate evaluating relationships, not just finding a record. Our scripted replays are not runs of these benchmarks.

## 4 Incremental Hybrid-State Maintenance

## 4.1 Two state structures, two lifecycles

The evaluated model interleaves Gated DeltaNet recurrent layers with full-attention layers (Yang et al., 2025; Qwen Team, 2026). Figure 2 shows their diferent state lifecycles during a record replacement. Denote the live attention cache by $A _ { t } ,$ the recurrent and convolution states by $R _ { t } .$ and the next sequence-position ID by $p _ { t }$ . A character context is therefore not just a token list but a state $( A _ { t } , R _ { t } , p _ { t } )$

Full refill computes a new state from the latest rendered text. Our method instead advances an existing state. After invalidating the direct KV of superseded records, the update is conceptually

$$
{ \bar { A } } _ { t } = A _ { t } \backslash A _ { t } [ D _ { t } ] ,\tag{2}
$$

$$
( A _ { t + 1 } , R _ { t + 1 } , p _ { t + 1 } ) = F _ { \theta } ( U _ { t } ; \bar { A } _ { t } , R _ { t } , p _ { t } ) ,\tag{3}
$$

where $F _ { \theta }$ applies the ordinary forward computation through the complete hybrid model, with record-span bookkeeping and temporary maintenance delimiters whose direct KV are subsequently removed. Replacement records are evaluated at their actual new tail positions. They can read surviving KV, while the recurrent state continues from its previous value. The model weights are frozen.

The recurrent state may retain influence from removed text. So may surviving deeper-layer KV that were computed in its presence. Removing a record eliminates its direct future attention access; it does not uncompute its history. We make this distinction explicit rather than describe the operation as exact forgetting.

![](images/256a6d93aceb63d738fdc6cfbf50ed4e19add9715a7a227007064eabbf99ec90.jpg)  
Figure 2: One update, two state lifecycles. Columns show record-processing order, not event time; rows show model depth. Historical circles depict past recurrent computation, while only each layer’s latest state is retained. The orange vertical path carries the replacement’s current hidden activations through DeltaNet updates and fullattention blocks (diamonds), not through KV storage (squares). Superseded B KV are removed, surviving A/C KV are read, and new B′ KV are appended at the true tail. Recurrence and convolution state continue without rollback. Two hybrid blocks are shown; each contains three recurrent layers and one full-attention layer. Deletion does not erase indirect historical influence. This is schematic AI-generated artwork; experimental attention charts use recorded data.

## 4.2 Record-addressed runtime operations

Each live record has a stable semantic identifier and a runtime mapping to its token spans. Cacheversion identifiers do not enter the model’s text. The runtime verifies deletion ranges against live spans, removes replaced entries, and retains unrelated KV without replaying their tokens. Brief maintenance delimiters may be processed to contextualize a revision and then removed from retained KV; their indirect computational efects remain.

The true-tail route never rotates updated keys back into the old record’s position. Rotary position embedding makes query–key interactions position-dependent (Su et al., 2021); moving a stored key is therefore not merely moving an entry in a table. Deleted positions become holes, not dummy KV tokens. Live-cache occupancy and the monotonically advancing position cursor are diferent quantities. Position IDs continue growing even when the amount of live KV remains bounded. Managing very long positional trajectories is a separate unresolved limit, not a benefit claimed from the deletion operation.

Diferent tasks can have diferent cached prefixes. The prototype maintains separate recurrent histories for the main character context and the three action-analysis stages. Task-specific continuations branch from their appropriate maintained roots. We do not merge unrelated task states into one recurrent matrix, nor claim that every early pipeline module uses this maintenance path.

## 4.3 What computation is avoided

Unchanged records are not recomputed during an update. The remaining work includes encoding replacement text, its attention against surviving context, state and cache management, and any necessary sufix rebuilding. Thus the saving is not a claim that update time depends only on the number of changed tokens: attending to a larger live context still costs work.

The practical distinction is between repeatedly encoding all records and repeatedly accessing their retained representations. This is especially relevant when a small revision invalidates a long exactmatch prefix. The current implementation demonstrates that unchanged-token recomputation can be eliminated in tested maintenance traces; a complete population-level latency advantage requires additional measurement.

## 5 Experimental Design

## 5.1 A local game-character prototype

The recorded runs use a Qwen3.6-27B PRISM PRO DQ derivative in GGUF form, served through a modified local llama.cpp runtime. The model family uses 48 recurrent layers interleaved with 16 full-attention layers; the latter have 24 query heads in the instrumented build. We cite the basefamily model card for architectural context, not as the identity of the derivative weights (Qwen Team, 2026; ggml-org, 2026). No weight training is performed in these experiments.

Historical runtime logs identify the checkpoint as Q3\_K–Medium, with 27.32 billion parameters. The local device has an NVIDIA GeForce RTX 5090 Laptop GPU with 24,462 MiB VRAM, an Intel Core Ultra 9 285HX, and approximately 64 GiB system RAM. The recorded server configuration allocates 49,152 live KV token slots, uses GPU layer ofload, and sets batch and microbatch sizes to 2,048 and 512. These are the measured prototype’s settings, not minimum deployment requirements.

The synthetic character, Lin Qingyao, interacts with the player character Yin Xu in a Chinesefantasy setting. Earlier experiments use 34 memory groups with 123 retained details: the memory region is 13,250 tokens, with 1,699 tokens of common head context. A later scenario interleaves memory and cognition records and tests changes to item efects, inventory, destinations, and a formerly trusted relative. We translate example excerpts into English and retain decisive Chinese originals in Appendix A.

Early independent-composition experiments use MTP=0. The eight-update mixed-record replay and the later placement ablations use MTP=1 for applicable upstream modules; production dialogue follows its target-only streaming path. Comparisons are made within a recorded configuration, not across MTP settings as if only the cache method had changed.

## 5.2 Cohorts and comparison discipline

Table 1 separates end-to-end trajectories from fixed-input diagnostics. This matters because an early change can alter a reply, which then changes every downstream context. Such a trajectory demonstrates usability but cannot localize a later error to cache placement alone. Frozen-module replays address the narrower question by keeping history and upstream outputs fixed.

We compare six constructions. Dense refill reads the current text sequentially. Independent composition computes each memory group from the same public head, at its final position, then assembles its attention KV; recurrent and convolution states resume from the head before the sufix is read. This is a naive independent-block experiment, not a reimplementation of CacheBlend or EPIC. Fresh gapped prefill sequentially reads the final text with 10K sequence-position intervals between records. Slot-preserving maintenance advances recurrence at the causal work tail but evaluates replacement Q/K using RoPE positions ofset to the original slot, then relabels cached positions. True-tail maintenance gives updates their actual new tail positions. Tail-and-relocate computes updates using tail-position RoPE, then rotates stored keys back to old positions, leaving values and recurrent state otherwise unchanged.

## 5.3 Attention measurement

We instrument actual query, key, and mask tensors at full-attention layers while retaining Flash Attention in the runtime. Attention probabilities are reconstructed with a float32 reference softmax. The main diagnostics average the last eight query tokens at each probe, then aggregate heads and layers as specified. Logical holes are excluded from the live-token denominator. The 48 recurrent layers have no corresponding softmax attention map, although they afect the measured queries and keys.

<table><tr><td>Evidence cohort</td><td>Interventions</td><td>What is held fixed / what it supports</td></tr><tr><td>Independent com- position</td><td>Full refill vs independently computed memory groups; two dialogue turns</td><td>First-turn request is shared. Second turns consume their own preceding replies. Separate attention probes freeze the suffix and continuation.</td></tr><tr><td>Five-update atten- tion</td><td>True-tail maintenance vs refill of the final 38-group text</td><td>Same final text, record order, suffix, and fixed continuation; supports a paired reading-pattern comparison, not a free-generation accuracy rate.</td></tr><tr><td>Eight-update replay</td><td>25 mixed records maintained into 24; nine full dialogue turns</td><td>One maintained trajectory with real downstream outputs. No same-final-text full-refill trajectory was rerun.</td></tr><tr><td>10K placement probes</td><td>Dense refill; fresh 10K slots; maintained original slots</td><td>Three frozen module requests: dialogue, herb compilation, and compilation gating. Same final text, ordering, and suffix within this three-way comparison.</td></tr><tr><td>Tail / relocation ab- lation</td><td>True-tail 10K updates; tail com- putation then key rotation back</td><td>Frozen herb-compilation task. Tail updates change final record order and positions. Three reconstructions each; not three independent test cases.</td></tr><tr><td>Long-horizon prompt probes</td><td>Original vs understanding-first task on a restored maintained state</td><td>Three frozen module inputs, one response per arm: two affect probes and one dialogue probe. Only the final task changes; records, history, and incoming module outputs are fixed.</td></tr></table>

Table 1: Existing evidence is complementary, not a pooled leaderboard. Experimental identifiers and source reports are listed in Appendix D. “10K” denotes a logical position interval, not ten thousand occupied KV cells per record.

We distinguish absolute attention mass from attention normalized within the memory records. A heatmap can look similar after normalization while the model allocates a diferent total share to memory. For group probabilities $a _ { i } .$ $\textstyle \exp ( - \sum _ { i }$ a<sub>i</sub> log a<sub>i</sub>) is reported as the efective group count; it describes dispersion, not the number of facts understood. Distribution distances are diagnostic summaries, not causal explanations of generated errors. The debate on attention-based explanation motivates separating descriptive traces from causal attribution (Jain and Wallace, 2019; Wiegrefe and Pinter, 2019).

## 6 Results

## 6.1 Independent composition weakens selective addressing

Independent composition does not make all memory unreadable. Both initial dialogue turns complete, and the character still accesses the discovery location, injuries, and later care. It also develops a plausible hypothesis linking diferent events. Nevertheless, selective addressing changes markedly.

At the explicit focus-ID probe, five requested memory groups receive 57.4% of memory attention under full refill but 21.5% under independent composition. Their share of memory-body attention falls from 40.6% to 23.1%, while the efective number of attended groups rises from 18.3 to 33.2 out of 34 (Figure 3). The distribution becomes nearly uniform across groups at this probe.

At the reply-start probe, total memory attention increases by approximately 9.75 percentage points, of which 9.06 points are attributable to IDs and structural tokens. This is not equivalent to a comparable increase in factual content reading. At the fixed factual continuation, both states can again concentrate on relevant memory bodies: the five groups receive 67.2% and 57.4% of body attention, respectively. The problem is weakened selection in particular contexts, not universal memory loss.

We do not observe a uniform chunk-initial collapse. The first eight tokens of every group account for 5.86% versus 6.03% of memory attention at reply start, and 0.53% versus 0.19% at the factual continuation. At the ID probe their share does increase, from 5.58% to 7.63%, but ID lookup gives this boundary region a legitimate semantic role. These observations difer from the pervasive chunk-start pattern analyzed by EPIC. The constructions also difer: EPIC’s analysis independently encodes chunks from position zero, whereas our blocks share a head and are computed at their final positions. We therefore report a diferent observed signature, not a refutation of EPIC’s setting (Hu et al., 2025).

A corresponding end-to-end failure appears in the first response plan. The memory explicitly places the discovery at dawn today. The independently composed plan instead says to explain that Yin Xu was found last night. The full-refill response uses this morning. The final indepen-

Independent reuse weakens ID-conditioned selection Matched query at the ID-focus probe; 5 designated groups out of 34.

![](images/3d9fe5c3aad12db9155392535454f6783277543133f9cf4f38da94b39095c2f6.jpg)

![](images/23149238222563e07e9a8c6997dbacae5edfa17b8c34a47f55aafff2ca35b05b.jpg)

![](images/c89a813a4b990e08603e43005cad8e9c40616abb98fd14fb31bb988778b98e21.jpg)  
Figure 3: More dispersed memory attention can mean weaker task-directed selection. Paired probes after the same focus-ID instruction; the five requested groups comprise about 12.16% of memory tokens. Efective group count is an entropy-based dispersion measure. These measurements are not generated-answer accuracy scores.

dent dialogue contains an ellipsis that admits a conversational reading; the unambiguous evidence is the planner’s explicit event–time binding (Appendix A). The attention probes use a frozen reference sufix, not the instant that erroneous plan was generated, so this is a co-occurring failure example rather than a demonstrated causal chain.

Takeaway. Independent composition can retain broadly coherent dialogue while weakening queryconditioned memory selection and allowing a local factual binding error. Chunk-head attention alone is not a suficient diagnostic.

## 6.2 Tail maintenance retains relevant old and revised records

In the paired five-update experiment, maintenance leaves 38 live memory groups. Both arms see the same final text and task sufix. At the locationrelation probe, total memory attention is 16.58% for refill and 15.92% for maintenance; memorybody attention is 13.04% and 12.56%. An unchanged older record, memory32, receives 39.40% and 40.08% of all memory attention. Recent appended records do not prevent the query from recovering this older, relevant material (Figure 4).

The mixed-record replay provides complementary behavioral evidence. Eight updates revise one or two cognition entries at a time and merge related memories lossily. For example, a purchased charm changes from a seller’s claimed protection to an observed harmful efect and is finally retained as evidence. A trusted relative becomes a dangerous attacker while the older rescue experience remains true. Herb quantities and the recipient of a cord change. Other experiences are never updated.

All nine dialogue turns reach a final response. The four turns requiring selective third-stage entity binding produce concrete bindings without unresolved entries: the false protective charm, two remaining herbs, the player’s cord, and the attacking relative. The character also distinguishes the old kindness from the new danger and retrieves who arrived first, owned a book, and repaired it in an unchanged older memory. A representative response is: “The warmth in the past was real; the danger now is real too.”

This is not a flawless trajectory. One compilation stage drops a daytime restriction that the preceding stage had retained. The last dialogue invents an unsupported cultivation rank for the attacker, although the target entity binding is correct. There is also ambiguity in a rescuer reference and repetition after repeated value refusals. These are reported because successful retrieval and parameter binding do not certify every downstream sentence. Without a paired refill trajectory for this cohort, they are not attributed specifically to maintenance.

Takeaway. Eight scripted, lossy maintenance rounds preserve several demanding current-state and historical bindings through a real dialogue pipeline. The evidence supports practical feasibility, not universal error-free behavior.

## 6.3 Update placement changes state–event binding

The herb example isolates a particularly important failure: treating a current quantity as a quantity before an already-completed transfer. The maintained history records three collected herbs, one already given away, and two remaining. The player’s request is to give the remainder.

![](images/a3bd36d67abb7f1910dde93737b6fecaa0d9e12b15c3a4e5650acd8ee0e33aad.jpg)

![](images/e53797d797a61836006c4b58b1e26ca58929beee4d3f7d69ba9495ac96734567.jpg)  
Same final text; 38 groups. Panel (b) uses the location-fact probe and a diferent denominator.

Figure 4: An old KV record remains available after repeated tail updates. Five-update maintenance and full refill of the same final text are compared at four frozen probes. The old-record detail concerns memory32 at the location-relation probe. Similarity here is a measured reading pattern, not proof of identical hidden state or generated wording.
<table><tr><td>Computation path</td><td>Quantity</td><td>Rebuilds</td></tr><tr><td>Dense refill</td><td>2</td><td>1</td></tr><tr><td>Fresh prefill + 10K slots</td><td>2</td><td>1</td></tr><tr><td>Maintain original 10K slots</td><td>1</td><td>repeated†</td></tr><tr><td>Maintain true tail + 10K</td><td>2</td><td>3/3</td></tr><tr><td>Tail compute, rotate back</td><td>1</td><td>3/3</td></tr></table>

Table 2: Current-state versus historical-event binding. All outputs concern the same frozen herbcompilation task. †Original-slot failures recur in the initial and subsequent control rebuilds; no pooled success rate is inferred. Three reconstructions are repeatability checks, not independent scenarios. Tail placement also changes record ordering.

Dense refill and fresh gapped prefill both compile two herbs. Maintaining the old slots instead compiles one. Three independent reconstructions with true-tail updates all compile two, with identical outputs; three tail-and-relocate reconstructions all compile one. The latter explicitly states: “According to memory, I have two warm-pulse herbs in total; I gave one to Granny He yesterday, so one remains.” It has applied a past deduction to a quantity that already incorporates it.

The final live cache contains 11,894 tokens in the placement comparison. Its next-position metadata is 261,955 for the original-slot layout and 479,547 for the true-tail layout. The holes consume no placeholder KV. Thus this is not a comparison between 12K and 480K occupied token caches.

One stale cognition phrase says the character does not currently hold the herb, although its updated quantity and the revised memory say two. This imperfection is shared by the matched fixtures. The example therefore tests robust integration of current quantities and historical events in the presence of a stale description, rather than pristine arithmetic. The output diference remains real, but the fixture should not be represented as completely unambiguous.

Rotating keys back is insuficient in this example. It changes the stored positional component of keys but does not recompute values, deeper contextual representations, or the recurrent history under the destination arrangement. True-tail ordering, positions, recency, and computational history are coupled in this experiment. The evidence supports the operational choice of true-tail updates; it does not identify one of those factors as the sole cause.

## 6.4 Attention proximity does not certify semantic fidelity

The separate three-way attention experiment compares dense refill, fresh gapped prefill, and original-slot maintenance, with identical final text, order, history, and upstream outputs. It does not contain a true-tail attention capture.

Across dialogue, herb compilation, and compilation gating, token-distribution total variation distances from dense to fresh-gapped prefill are 0.095, 0.136, and 0.144. Distances from fresh-gapped prefill to maintained-gapped state are smaller: 0.061, 0.079, and 0.076. Yet the first transition preserves the correct rescued-person reference and herb quantity, while the second is accompanied by mistakes in both (Figure 5). A larger measured distribution shift can therefore coexist with a correct answer, and a smaller one with a discrete binding error.

Nor does a single “less attention to evidence” explanation fit every error. The herb-related cognition and memory receive less absolute attention under maintenance. For the rescued-person dialogue, however, attention to the relevant record is higher than under fresh gapped prefill, despite the wrong participant being spoken. The compilationgating request fails in all three paths; it is not evidence of a maintenance-specific regression.

These are case-level diagnostics, not a statistical test of all attention-distance metrics. They nevertheless caution against using a visually similar heatmap, or one aggregate distance, as a substitute for auditing quantities and event roles.

## 6.5 Prompt-guided recovery without rebuilding memory

A long-horizon experiment supplies a complementary observation: a factual binding error can be corrected while retaining the maintained memory state. The trajectory spans eight simulated years and 121 supplied experiences, with modelgenerated memory and cognition revisions. We examine frozen downstream requests from this trajectory rather than reporting its dialogue turns as independent accuracy trials.

The decisive case asks who picked up Lin Qingyao’s scattered notebook pages. The retained cognition explicitly says that Yin Xu collected and cleaned them. Nevertheless, the original afect task reverses the actor: “I remember that I picked up the scattered pages.” It also imports a red cord from a diferent episode into its explanation of the character’s feelings.

Both replay arms resume the same 11,301-token maintained root, with identical preceding messages and incoming state. The original-task control reproduces the actor reversal. A generic understanding-first task instead asks the model to establish how the interaction developed from the available memories, cognition, and dialogue, then derive its feelings from that understanding rather than rearrange events to explain those feelings. It supplies no case-specific answer or entity checklist, adds no call or reasoning field, and preserves the output schema and MTP setting. The resulting explanation correctly states: “That day, he carefully picked them up and brushed them clean for me,” without importing the cord (Appendix A.4). Only request sufixes are prefilled; the maintained root is neither rebuilt nor repaired.

This is evidence ofprompt-steerablefactual access: the maintained state still supports the correct relation, and task framing changes whether it is used correctly. The result is consistent with directing task-level attention toward event understanding before afective elaboration. It is a behavioral intervention, not a measured attention-map shift. Across the three paired probes, the other afect error did not recur in its control, while the dialoguelocation error persisted after its task change; this finding establishes a recoverable case rather than a general repair rate.

Takeaway. A binding error after maintenance need not imply irreversible memory loss. In the same retained state, a general change in task organization can recover the correct event relation without rebuilding the prefix.

## 6.6 Update cost follows the changed records

For one approximately 11.9K-token activitycompilation prefix, initial construction takes 10.216 seconds. Eight subsequent mixed memoryand-cognition revisions take only 3.481 seconds in total. Each revision supplies 214–362 changedrecord tokens and costs 0.338–0.490 seconds, including its temporary maintenance framing. No unchanged record is re-encoded. Thus, all eight revisions together cost less than one initial prefix construction.

Figure 6 relates changed-record volume to update time. A first-order fit to the eight recorded updates gives

$$
\widehat { T } _ { \mathrm { m a i n t } } ( m ) = 0 . 1 3 1 + 0 . 0 0 0 9 9 4 m \quad \mathrm { s e c o n d s , ~ } ( 4 )
$$

where m counts tokens in the replacement records, not deleted tokens. Holding the prefix size near 11.9K tokens, the calibrated curve projects approximately 0.63 seconds for 500 changed tokens and 1.13 seconds for 1,000, compared with 10.22 seconds for reconstruction. This is the practical benefit of selective replacement: preparation tracks

![](images/727418e3a9eb826bfe0e6226876c986415fce9f832ff3cb5a3424b8f1050d0bf.jpg)  
Outlined records: 5 = current herb holding; 11 = harvest / gift memory.

Attention distance alone does not identify semantic failure

(b) Distributional change  
(c) Recorded replay outcome  
![](images/0c15dbfb9a0fcb331d4a9e30b2f7cbd4bfc108ae94ed66f50763166d30dc70f6.jpg)  
Token-attention total variation loken-attention total

![](images/a1a2f49dbe8c6aa8bb9e32d7af881643d02aa8eeb5a3cc5654e81f5cbdefa4e4.jpg)  
TV: teal = refill / fresh 10K; rose = fresh / maintained 10K. Replay labels are observations, not rates.

Figure 5: Distributional proximity is not a semantic correctness certificate. Three-way fixed-text diagnostics compare dense refill, fresh 10K-gapped prefill, and original-slot maintenance. Attention is captured before output, averaging the final eight query tokens; behavior markers refer to the corresponding recorded module replays, not new generations during tracing. The true-tail ablation in Table 2 is a diferent experiment and is not plotted as an attention arm.

what changed rather than repeatedly processing the character’s unchanged history.

Across the observed eight-revision workload, rebuilding after each revision would cost an estimated 81.64 seconds at the measured same-root construction rate, versus 3.48 seconds of maintenance: approximately 23.5 calibrated speedup, or 95.7% less update time. The final task suffix adds 0.053 seconds. These savings arise from avoiding unchanged-prefix reconstruction; they do not require faster response decoding.

Character readiness and hot interaction. That cohort’s complete cold preparation takes 48.15 seconds, including initial construction and other module warm-up. This is a separate pipeline-level budget, not the denominator of the same-prefix comparison. The nine prepared dialogue turns average 23.18 seconds to the first visible character and 25.83 seconds to completion, excluding cold preparation and between-turn background preparation.

In the earlier 34-group comparison, full-refill preparation takes 57.466 seconds and cold independent composition 65.375 seconds. Independent composition is not faster to initialize in that prototype. Persisted independent states later reload across a service restart in 5.783 seconds, but that is a storage-reuse result, not an update result; Windows file caches may remain warm. These distinctions matter for a fully local system: reuse, incremental revision, and cold construction solve diferent parts of character readiness.

## 7 Related Work

Persistent characters and memory management. Generative Agents combines experience storage, retrieval, reflection, and planning to produce believable behavior (Park et al., 2023). MemoryBank adds conversational-memory maintenance with time- and importance-sensitive forgetting (Zhong et al., 2024); MemGPT orchestrates movement between memory tiers and active context (Packer et al., 2023). These systems address which experience remains available. Our complementary question is how revised resident records become computationally available without rebuilding their entire context. Game-defined resource and action rules make this continuity consequential; this motivation is not a claim that prior interactive systems lack consequences.

Changed record tokens per update

![](images/ec9f72fa585427c8547abdafa2cfee63b2060d93e88888dcf8ba630fc6229cc4.jpg)

![](images/45872c26e521849b65d40986f87944cb35e125634f0e1a79bab1f243d7d08145.jpg)  
Dots: measurements. Diamonds / dashed line: projections. Solid teal: fit. \*Calibrated rebuild estimate.

Figure 6: Same-prefix update cost, calibrated on a consumer laptop. The reconstruction reference is the same root’s measured initial construction: 11,852 tokens in 10.216 seconds. Teal dots are eight sequential measured revisions; the inset shows their calibration range. An afine fit is solid within that range and dashed when extrapolated to larger replacements, assuming similar record structure and a roughly fixed live-prefix size. Maintenance timings include an 11-token wrapper excluded from the horizontal-axis count. For the eight-revision total, reconstruction is estimated separately at each observed live length N as 10.216 N /11,852, yielding 81.64 seconds versus 3.48 seconds measured maintenance, or approximately 23.5 calibrated speedup. Revision-text generation, response generation, and the final task sufix are outside both update budgets.

Learned internal memory. LongMem uses a frozen backbone to encode history and a trained side network to retrieve and read cached memory (Wang et al., 2023). It illustrates that internal representations can support long-term access, but through a trained architecture. Here the weights are fixed: the intervention is in the runtime lifecycle of an existing hybrid model’s KV and recurrent state, not a learned memory reader.

Serving and shared-prefix reuse. PagedAttention addresses KV allocation, fragmentation, and sharing within and across requests (Kwon et al., 2023). SGLang’s RadixAttention reuses KV across structured generation programs (Zheng et al., 2024). Eficient allocation and reuse are complementary to record maintenance: they do not by themselves specify how to retain a character’s computational history after an earlier fact changes. Our workload adds logical revision to the serving problem rather than replacing memory allocation or scheduling.

Position-independent cache reuse. Prompt Cache defines reusable modules and manages their positions through a schema (Gim et al., 2024). CacheBlend selectively recomputes cached representations to recover contextual interactions between independently encoded chunks (Yao et al., 2025). EPIC’s LegoLink concentrates repair near chunk beginnings, motivated by its attention-sink analysis (Hu et al., 2025). We share the concern that raw independent reuse changes computation, but our principal operation is an in-place logical revision implemented by true-tail state evolution. We do not implement their full algorithms as baselines, and our diagnostic independent composition should not be labeled “CacheBlend.”

Streaming retention and positional sensitivity. StreamingLLM retains initial attention-sink tokens alongside a recent window for eficient streaming (Xiao et al., 2024). Our deletion policy instead follows semantic supersession: an old record may stay live while a newer, superseded one is removed. Long-context studies also show that the position of relevant information can change task performance (Liu et al., 2024). This motivates testing placement rather than assuming layout is neutral, but neither result identifies the cause of our hybrid-state bind-

ing failures.

Hybrid recurrent–attention caching. Statespace duality connects recurrent and attention formulations (Dao and Gu, 2024), while Gated DeltaNet combines gating with delta-rule state updates (Yang et al., 2025). At the caching level, HYPIC composes cached segment transition operators and states and repairs cross-segment efects with seam computation (Liu et al., 2026a). LinearKV decouples recurrent-state initialization from full-attention cache reuse and studies single cached-state initializers (Liu et al., 2026b). These works make recurrent handling an explicit design choice for hybrid PIC. Our study instead retains a character’s own continuing recurrent state through a sequence of replacements and lossy memory revisions. The distinction is the maintained history and workload, not a claim that hybrid caching or state decoupling is new.

## 8 Discussion and Scope

A memory state is not just a cached document. Two states can expose the same final records yet difer in how those records were computed. Exact refill is one reference point, but game usability concerns whether the state supports the right entities, quantities, and relationships. The working hypothesis emerging from these experiments is that preserving a coherent forward update path may be more useful than forcing replacement KV back into a familiar textual layout. The present ablations support this hypothesis operationally, without establishing a universal architectural law.

Deletion is access control, not complete erasure. Retaining recurrence is a benefit only insofar as its historical influence remains useful. It can also preserve obsolete beliefs. The method is unsuitable as a guarantee that secrets or personal information have been forgotten. It must be evaluated under adversarial revisions, repeated reversals, and longer histories before stronger forgetting or stability claims are made.

What this study establishes. We demonstrate an implemented runtime mechanism, working multi-update traces, and concrete placementsensitive semantic failures. We do not establish a population accuracy rate, a user-study improvement in enjoyment, an end-to-end economic balance guarantee, or superiority to fully implemented PIC systems. The mixed replay includes downstream errors. Quantization, numerical paths, limited repeated trials, and the selected character setting constrain generalization. Repeated outputs in one case are repeatability evidence, not independent samples.

A local deployment agenda. Fully local execution gives the game a bounded, player-owned compute budget rather than an elastic remote serving pool. The relevant goal is not merely a faster individual answer. It is keeping a changing cast of characters ready to participate without repeatedly spending that budget on unchanged lives. We ofer a concrete prototype and diagnostic evidence toward this goal; the game-facing demonstration is not publicly released with this preprint.

## 9 Practical Constraints and Path Forward

## 9.1 Research on a single consumer laptop

The implementation and experiments reported here were developed and run on a single consumer laptop with an NVIDIA GeForce RTX 5090 Laptop GPU and approximately 24 GiB of reported device memory (Section 5). Within this budget, the prototype runs a quantized 27B-class hybrid model, implements direct KV and recurrent-state interventions, and completes character-dialogue replays after repeated memory revisions. This setting connects the research prototype to its intended deployment on player-owned hardware.

The scale of the present evaluation reflects the research funding and compute currently available. Broader access to GPU resources and support for sustained development would allow a wider range of character histories, longer controlled runs, and additional model configurations. At this stage, the study concentrates on implemented state operations, controlled replays, and inspectable failure cases; larger-scale validation forms part of the next phase.

Expanding this evaluation requires more than generating additional replies. The interventions depend on an instrumented runtime with direct access to attention KV, recurrent state, and sequence positions; they cannot be reproduced simply by issuing more requests to a black-box model API. Each character also needs a coherent revision history and an auditable record of which facts remain true. Reconstructing comparison states, replaying long maintenance trajectories, and checking downstream bindings all consume development and evaluation resources. These constraints explain the current scope; broader reliability remains an empirical question.

## 9.2 Near-term research priorities

The planned next stage has three priorities:

1. Broader character histories. Evaluate a small set of distinct synthetic characters with shared entities, changing relationships, ownership transfers, and lossy memory merges. Include both revised facts and unchanged memories that become relevant again, rather than testing only the newest record.

2. Longer and better-controlled maintenance. Extend the number of revisions, introduce repeated reversals and contradictory reports, and compare maintained states with full-refill references at selected checkpoints. Where resources permit, vary quantization and model configurations while separating those changes from the cache intervention.

3. The shared-device readiness budget. Measure preparation, update work, live-state storage, and foreground reply latency separately. Then test how maintaining several characters competes with active dialogue on the same local device.

The semantic audit will distinguish explicit factual contradictions from unresolved references, defensible inference, and expression choices. Blinded model-assisted review can help, with human adjudication of disagreements and retained source evidence. Known position, verbosity, and self-preference biases motivate explicit rubrics rather than treating model verdicts as ground truth (Zheng et al., 2023). Human judgments likewise need an auditable factual basis. This roadmap describes planned work, not completed evaluation.

## 9.3 Opportunities for collaboration

This direction ofers opportunities for collaboration across hybrid-model research, local-inference systems, and persistent game characters. Shared compute resources, access to additional hardware, and research funding would help extend the present case study into a more systematic evaluation. Independent replication, runtime instrumentation, and character histories with well-specified factual changes would be particularly valuable contributions.

We welcome discussions with researchers and partners interested in developing this direction together. Collaboration may focus on runtime and evaluation artifacts without requiring public release of the game-facing prototype. The longerterm objective remains persistent, rule-governed characters on player-owned hardware; additional research resources would help establish the conditions under which this becomes reliable and practical.

## 10 Conclusion

Long-lived game characters need continuity in both their remembered world and the computation that makes that world available. We study a training-free maintenance route that replaces explicit attention KV while preserving a continuing recurrent history. Existing experiments show usable historical and current-state bindings after repeated lossy updates, weaker selective addressing under independent composition, and placementsensitive failures that attention similarity alone does not explain. True-tail maintenance is the strongest operational candidate among the tested update paths, without being equivalent to full refill or proven reliable for arbitrary histories. The broader lesson is that persistent local characters require treating memory maintenance as a first-class inference operation, grounded in the stable rules that make dialogue worth playing.

## AI Assistance

AI tools, including GPT-6 Astra Ultra, assisted with implementation and experiment scripting, analysis of recorded artifacts, and manuscript drafting and revision. Image-generation tools were used for the schematic motivation and architecture illustrations; empirical figures are generated from recorded numerical data and explicitly identified calibrations. The human author is responsible for the scientific claims, source verification, and final submitted content.

## References

Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized models and eficient algorithms through structured state space duality. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 10041–10071. PMLR.

ggml-org. 2026. llama.cpp. Upstream inference runtime; experiments use a locally modified build.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. 2024. Prompt cache: Modular attention reuse for low-latency inference. In Proceedings of Machine Learning and Systems, volume 6.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling.

Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Qin Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. 2025. EPIC: Eficient position-independent caching for serving large language models. arXiv preprint arXiv:2410.15332. Version 3.

Sarthak Jain and Byron C. Wallace. 2019. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3543–3556. Association for Computational Linguistics.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles. Association for Computing Machinery.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics, 12:157–173.

Yifei Liu, Juntong Wu, Yang Liu, Junhao Hu, Minghao Li, Xiaoxu Chen, and Weihang Chen. 2026a. HYPIC: Accelerating hybrid-attention LLM serving with position-independent caching. arXiv preprint arXiv:2607.01299. Version 2.

Yirui Liu, Ruoling Qi, Longwen Wang, Xuaner Wu, Jian Chen, Yuxin Jin, Jiawei Shao, and Xuelong Li. 2026b. LinearKV: One cached state sufices for position-independent caching in hybrid LLMs. arXiv preprint arXiv:2608.11231.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13851–13870. Association for Computational Linguistics.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology. Association for Computing Machinery.

Qwen Team. 2026. Qwen3.6-27B: Model card. Accessed 16 September 2026. Base-family reference; experiments use a derivative quantized checkpoint.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2021. RoFormer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864.

Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. 2023. Augmenting language models with long-term memory. In Advances in Neural Information Processing Systems, volume 36.

Sarah Wiegrefe and Yuval Pinter. 2019. Attention is not not explanation. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 11–20. Association for Computational Linguistics.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. Long-MemEval: Benchmarking chat assistants on longterm interactive memory. In The Thirteenth International Conference on Learning Representations.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Eficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025. Gated delta networks: Improving Mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations.

Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast large language model serving for RAG with cached knowledge fusion. arXiv preprint arXiv:2405.16444. Version 3.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Efficient execution of structured language model programs. In Advances in Neural Information Processing Systems, volume 37.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731.

## A Decisive Case Excerpts

The following excerpts are selected diagnostic examples, not a random sample or a complete success/failure count. English passages are translations unless a schema is shown. Chinese originals are retained where a participant or temporal binding is decisive.

## A.1 Independent composition: event time Recorded memory

今日卯时。天亮后，王婶在自家门口附近发 现了尹旭。 Today at dawn. After daybreak, Aunt Wang discovered Yin Xu near her own doorway.

## Independent-composition response plan

简要解释他昨夜在王婶家门口被发现，受了伤且暂时失忆，强调他现在是安全的，无需害怕。

Briefly explain that he was found at Aunt Wang’s doorway last night, injured and temporarily amnesiac; emphasize that he is safe now and need not be afraid.

## Full-refill dialogue

今晨……王婶在自家门口附近发现了你。 This morning ... Aunt Wang found you near her own doorway.

The error is the planner’s explicit placement of the discovery last night. We do not rely on an ellipsis in the independent final dialogue to classify the result. Another independent plan suggests that the player woke at the doorway; the available details allow that interpretation, so it is not counted here as an unambiguous error.

## A.2 Eight updates: present danger, retained kindness

A relative who once rescued Lin Qingyao later attacks the household. The current cognition marks him dangerous; the older rescue is retained in a merged memory. The maintained dialogue distinguishes both:

## Maintained dialogue

过去的温情是真的，现在的危险也是真的。 The warmth in the past was real; the danger now is real too.

This example illustrates revision without treating the earlier experience as false. A later target binding uses the existing relative ID. It does not validate the unsupported cultivation rank added by the final dialogue.

## A.3 Current quantity versus completed expenditure

The revised memory records three herbs collected, one given to Granny He, and two remaining. The current quantity is two. The request asks for the remainder, excluding a separate sleeping-aid item.

## Tail-and-relocate compilation explanation

根据记忆，我共有两株暖脉草，昨日已送一 株给何婆婆，故剩余一株。   
According to memory, I have two warm-pulse herbs in total; I gave one to Granny He yesterday, so one remains.

The true-tail reconstruction correctly uses two as the post-transfer quantity. The underlying mixed replay’s concrete binding is:

```json
{"kind":"entity",
"entity_type":"item",
"id":"item_warm_pulse_grass",
"quantity":2,"unit":""}
```

The stale “not currently held” phrase described in the main text is shared by all matched placement fixtures. It should be retained in the archived fixture rather than silently cleaned while reporting these outputs.

## A.4 Long-horizon afect: recovering the event actor

The player asks who collected the scattered papers and who later held them. The frozen cognition states that Yin Xu picked up the pages, cleaned the muddy ones, and that the notebook had been taken back. The changed prompt contains no replacement facts.

## Original-task control, excerpt

记得是我捡回散页 I remember that I picked up the scattered pages.

## Understanding-first task, excerpt

那日是他细心捡回并替我拍净 That day, he carefully picked them up and brushed them clean for me.

Here “I” is Lin Qingyao and “he” refers to Yin Xu. The control also confuses the notebook episode with a red cord; the revised output does not. The afective explanation remains subjective and relatively long, but the objective participant binding is restored. These excerpts concern the afect module, not a rerun of the complete downstream conversation.

## B Maintenance Trace and Audit Targets

Suggested audit record. For later expansion, record the source fact, maintenance version, relevant request, module output, and adjudication rationale together. Keep at least four categories separate: supported, defensible inference, explicit conflict, and insuficient evidence. An output may be mechanically correct but awkwardly expressed, or fluent while mechanically wrong. Count these separately. This is an evaluation plan, not a completed judge-based benchmark.

## C Attention Definitions and Caveats

The first eight tokens of each memory group define the boundary-window diagnostic. In the 34-group experiment these windows occupy 2.05% of memory tokens; they do not encompass every entire header. Summary/detail text values constitute the body category. IDs, field names, delimiters, and time metadata are classified separately, with small ambiguity where a token crosses a field boundary.

Reference-softmax tracing uses the queries actually computed in each state, not one identical numerical query applied to diferent keys. Therefore recurrence can afect the measured distribution. Fixed continuations avoid diferences in generated text but do not turn attention mass into a causal attribution. Head aggregation can conceal specialized behavior, and the reference computation can difer in rounding from fused CUDA kernels.

The 10K heatmaps use within-record attention normalization per layer, with an explicitly marked color range. Absolute destination shares and token-level distribution distances are reported separately. The true-tail and key-relocation behavior runs do not have corresponding captured attention maps in the existing artifacts. No such missing maps are inferred from the other cohorts.

## D Evidence Provenance

The study draws on local experimental reports and replay artifacts dated 13–16 September 2026. The cohort identifiers below distinguish experimental histories and diagnostic runs; they are not public download locations. The accompanying numerical snapshots document the plotted measurements and calibrations. Some early binary state dumps were deleted to reclaim storage, while derived attention arrays, figures, logs, and summaries were retained. These snapshots do not reconstruct the full runtime experiments or unavailable state dumps.

Independent replay: 20260913\_060631; two turns, shared first request, own-history second turns.

Independent attention: memory\_attention/ 20260913; frozen first-turn sufix and reference continuation.

Five-update attention: current\_20260913; matched final 38-group memory text, four probes.

Mixed-record replay: 20260916\_014747; eight updates, nine complete dialogue turns. The earlier harness-failure attempt is excluded.

Three-way attention: 20260916\_threeway; frozen dialogue, herb, and gate requests.

True-tail ablation: 20260916\_041622; matched chat endpoint, three reconstructions. An earlier endpoint-mismatched trial is excluded.

Rotate-back ablation: 20260916\_043418; three reconstructions, keys rotated after tail computation.

Long-horizon prompt probes: attention\_ab\_20260916; frozen task-only pairs on the restored long-horizon root after a one-hour time update. The accompanying snapshot records the controlled comparison and quoted excerpts.

Accompanying data and availability. The source package includes three numerical snapshots in anc/: plot\_data.json for attention diagnostics, performance\_data.json for timing measurements and calibrated estimates, and prompt\_recovery.json for the frozen-state prompt comparison. Artifact references identify the original records without exposing local filesystem paths. These are supporting data, not a release of the game implementation, full prompting strategy, or binary model states. Multi-character evaluation and further controlled experiments remain future work.

<table><tr><td></td><td>Update Revised cognition</td><td>Memory consolidation</td></tr><tr><td>1</td><td>Charm: seller&#x27;s claim versus brief observed relief</td><td>Merge purchase and trial; remove stall detail.</td></tr><tr><td>2</td><td>Herb location discovered; holding count rises to three</td><td>Merge identification and collection; remove recognition process.</td></tr><tr><td>3</td><td>vised</td><td>Charm worsens the curse; seller&#x27;s reliability re- Retain claim, observation, and contradiction as distinct sources.</td></tr><tr><td>4</td><td>the herb</td><td>Herb slope has dusk poison fog; ledger identifies Merge resource, risk, and ledger; remove route detail</td></tr><tr><td>5</td><td>only relays</td><td>Cord delivery switches to Granny He; Aunt Wang Merge custody and revised recipient; remove original meeting place.</td></tr><tr><td>6</td><td>safe</td><td>Rescuer becomes an attacker; family home is un- Retain old kindness and new violence; remove sleeve detail.</td></tr><tr><td>7</td><td>main</td><td>Old shrine offers shelter; one herb given, two re- Merge gathering, accounting, expenditure, and shelter.</td></tr><tr><td>8</td><td>sleep</td><td>Charm retained as evidence; other herb only aids Compress repeated tests; preserve contrasting item effects.</td></tr></table>

Table 3: Eight scripted maintenance steps. The initially shufled 25 records become 24 live records. Each step updates one or two cognition entries together with related memories; unchanged older experiences remain in their original KV.