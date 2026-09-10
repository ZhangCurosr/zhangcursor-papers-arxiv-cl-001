# Contrastive Projection: Reading Transformer Internals by Differencing Logit Lenses

Olli Tuomi Evident Solutions Oy

## Abstract

Reading a transformer’s internal states in token space is easy to do and hard to trust: a logit lens on a single hidden state is dominated, at intermediate layers, by the generic tokens the model would predict for almost any input. We read the difference instead. Subtracting two closely matched prompts’ hidden states and projecting through the unembedding cancels the shared component and surfaces what separates them, an operation equivalent to reading a RepE/ActAdd steering vector through a logit lens. Built into a training-free tracer that reads at every position, sub-layer, and head and averages over designed baselines, it traces a compoundnoun MLP→attention chain in Phi-2, confirmed there by activation patching, with the same distinction recovered across three architectures by readout and probe rather than by patching; it reads what retrieval surfaces for real versus fictional entities, and reads metaphor as a set of domain-to-domain mappings rather than a single figurativity feature. A cross-seed control marks the boundary: across five networks differing only in initialization, the same distinction surfaces as almost entirely different tokens (top-10 overlap 0.08). What a computation looks like in token space is network-specific; the distinction it draws is not.

## 1 Introduction

A transformer processing “The hot dog was” predicts continuations about food. The same model given “The cold dog was” predicts continuations about an animal. The logit lens (nostalgebraist, 2020) projects each hidden state through $W _ { U }$ . But the single-input projection is dominated by strong shared signals. At intermediate layers it reads the same high-frequency function words for both inputs (“not, no, made, more”), not the food or animal content (Table 1, first column).

Subtracting one hidden state from the other before projecting through W<sub>U</sub> cancels the shared signals and reads what differs. The hot dog−cold dog difference reads food vocabulary from layer 8 on (“fried, crispy, delicious, flavor”), absent from either constituent’s logit-lens top-20 (0–1/5 overlap through L24). The same contrast read the other way, cold−hot, surfaces animal vocabulary (“grooming, Paw, Bark, breeds”), the pet reading that the food direction cancels. The difference is signed: each direction reports what one side carries that the other does not.

Two closely matched inputs share most of their computation, so what survives the subtraction is

the component where the model treats them differently.
<table><tr><td>L</td><td>Raw (both)</td><td>hot - cold</td><td>cold – hot</td></tr><tr><td>8</td><td>nt, no, party</td><td>fried, hors, crispy</td><td>vet, Pent, ogie</td></tr><tr><td>22</td><td>not, made, more</td><td>tast, delicious, flavor</td><td>grooming, euth, Paw</td></tr><tr><td>24</td><td>not, more, so</td><td>delicious, tast, flavor</td><td>grooming, Paw, Bark</td></tr></table>

Table 1: The opening contrast, read at the final token of “The hot dog was” versus “The cold dog was” (Phi-2). The raw logit lens of either input reads high-frequency function words at every layer; the hot−cold difference reads food vocabulary (legible from L8), the cold−hot difference reads animal vocabulary (sharpening by L20). Verbatim top-3; some entries are sub-word fragments (“nt,” “hors,” “euth”) or noise (“Pent, ogie”).

Every contrastive reading is pair-specific. It reports what separates one particular pair of prompts. A tight pair that varies one thing reads more cleanly than a loose pair that varies several. Contrast design steers the reading: what it surfaces is chosen, not discovered. This is a mechanistic form of contrastive explanation, where one explains why one case rather than a chosen foil, and the foil fixes the answer (van Fraassen, 1980; Lipton, 1990).

The primitive is not new. Differencing two prompts’ hidden states and projecting through $W _ { U }$ is the arithmetic of a Representation Engineering (RepE) or Activation Addition (ActAdd) steering vector (Zou et al., 2023; Turner et al., 2023) read through a logit lens (nostalgebraist, 2020). Du et al. (2026) already used this exact operation. They decoded an activation difference through the logit lens to trace control signals for reflection in R1-style reasoning models. Our contribution is what this readout becomes as a systematic tracer. Read at every position and sub-layer, and averaged over designed baselines, it locates where an axis of variation first becomes legible in token space, which sub-layer writes it, and which head carries it. We add three techniques and a set of applications:

1. Systematic trajectory reading. Per-position tracing locates where a distinction first appears and how it flows between positions. Per-head decomposition identifies which attention heads carry it. Layer-by-layer readout tracks how the content changes from early detection to final prediction.

2. Multi-contrast triangulation. A single pair reports every way the two prompts differ, not only the intended axis, so its readout can be hard to read. Contrast the target against several baselines that share the intended axis but differ on the incidental ones, then average. The incidental axes cancel and the shared one remains (§2.7).

3. Contrast design. What the readout surfaces depends on how the pair is built. We give rules for matched contrasts: a shared preamble to avoid the massive-activation first token, a matched current token at the read position so the difference is content and not token identity, and a matched predicted next token to expose mid- and late-layer computation (§2.5).

We apply the method to Phi-2 (2.7B). We re-run the compound-noun circuit across three architectures (Phi-2, Pythia-1.4B, and Qwen2.5-1.5B) to separate what generalizes from what is model-specific (§3.2). The cases shown are selected from several hundred readings. We report the most illustrative.

One caveat frames everything that follows. The specific tokens we read out (“fried,” “Nepal,

Tibet,” “delicious”) are illustrative, not a stable code. A cross-seed control (§6) shows that five networks differing only in initialization surface the same distinction as almost entirely different tokens (top-10 overlap 0.08), while the distinction itself holds in every seed. Throughout, read a token as evidence of what a contrast separates in one network, not as a canonical vocabulary.

## 2 Method

## 2.1 Contrastive projection

Given two inputs c and k:

1. Run both and extract hidden states at every layer at the read position: $h _ { c } [ L ]$ and $h _ { k } [ L ]$ for $L = 0 , \ldots , N .$

2. Compute $\Delta h [ L ] = h _ { c } [ L ] - h _ { k } [ L ]$

3. Project: $\Delta \mathrm { l o g i t s } [ L ] = \Delta h [ L ] \cdot W _ { U } ^ { \top }$

4. Read the top-K most positive tokens (associated with input c) and the top-K most negative tokens (associated with input k). The two poles together describe the contrast in token space.

No parameters are fitted. The choices are the input pair, the read position, and K. The pair should be aligned: the same token length, read at the same index, so position-dependent structure (strong under rotary embeddings) cancels in the difference rather than leaking into the readout. Section 2.5 shows this matters most at the first token, which carries a massive activation. Where names of different lengths make exact alignment impossible (§5), we do not simply trust the read; we verify it against a length-matched control.

Two practical notes. We project through the logit lens $( W _ { U } )$ , not the input embedding. The residual stream aligns with output space, and projecting the same states through $W _ { E }$ returns noise. We also project the raw states, bypassing the final LayerNorm $( \mathrm { L N } _ { f } )$ . For a difference of two matched states the learned shift cancels. The per-dimension rescaling preserves the top-5 tokens (mean cosine ≈ 0.95 against the normalized projection) and reshuffles only the ranking tail, which does not affect an axis-level reading. Folding the $\mathrm { L N } _ { f }$ gain into $W _ { U } ,$ , as TransformerLens does, is equivalent.

## 2.2 Per-position reading

By reading the contrast at every position, we trace information flow: at which position does a meaning distinction first appear? Does it appear at the differing token itself, or at a later position that attends to it?

## 2.3 Attention and MLP decomposition

Each transformer layer adds two components to the residual stream: attention output and MLP output. We capture both via forward hooks and project each separately through $W _ { U }$ . This reads which of the two writes a given content distinction at each layer, taken directly from the projection with no trained probe.

Projecting MLP writes and attention-head writes through $W _ { U }$ is old news: the logit lens has been applied per-component before us. What the contrast adds is that differencing two matched runs often exposes structure inside a component that the raw per-component read does not show. Where a single run’s MLP or head write projects to token soup, the difference of two runs at the same component can resolve into a legible distinction, because the shared bulk cancels and the part that separates the two inputs is left.

## 2.4 What the projection reads

The contrastive projection reads the difference of the two states through $W _ { U }$ . A raw logit-lens readout of either input is dominated by signals the two inputs share (often high-frequency function words at the top of the ranking). The subtraction cancels that shared component and reads what remains.

$W _ { U }$ projection assigns token labels to directions in the residual stream. Whether those labels are interpretable varies with the contrast and the layer. For a well-chosen contrast at a legible layer they name what differs between the two inputs. Many other projections return token soup (§2.7). The readout is a projection onto $W _ { U }$ , not a decomposition of the computation. A component of the difference that aligns with no token direction does not appear, and its absence from the readout is not evidence of its absence from the state (Tuomi, 2026). Our claim is limited: for well-chosen contrasts, part of what differs projects to legible tokens.

## 2.5 Contrast design

What the readout surfaces depends on how the pair is built. Three rules make a contrast legible: a shared preamble, a matched current token, and a matched predicted next token.

Preambles. Preamble is our name for the shared text before the contrasted change. Early tokens are a poor place to put a contrast, for two reasons. The sharp one is measurable: the first token carries a massive activation, residual norm about twenty times the rest (941 versus ∼45 at L8), the attention-sink structure of Sun et al. (2026). A difference on that token is amplified and dominates the read, which is why the capitalisation mismatch below is so destructive. The subtler one does not show up as norm. In its first several tokens the model is still settling into the register and format of the text. Whatever the cause, a difference placed this early reads less cleanly even when its norm is ordinary: our bare prompts align with the preambled read at only cosine 0.85 (Table 2), and shorter prompts read worse. So the rule is to place the contrasted change several tokens in, where the read is clearest. A shared lead-in of a few plain tokens does this.

A first-token mismatch shows why the frame matters (Table 2). Read at the final token, the matched pair “The hot dog was” / “The cold dog was” reads food at L20 (top token “tast,” food at rank 3), aligning with the preambled read at cosine 0.85. Lowercasing only the second prompt’s first token, “The hot dog was” / “the cold dog was,” pushes food down to rank 12 and puts non-food tokens on top $( ^ { \prime \prime } { \mathrm { s o l d } } , \mathrm { m o r e } ^ { \prime \prime } )$ , dropping the alignment to cosine 0.64. A shared preamble reads food most stably (rank 0). The same edibility contrast is read along different directions depending on the frame.
<table><tr><td>First prompt</td><td>Second prompt</td><td>Readout at L20</td><td>food</td><td>COS</td></tr><tr><td>The hot dog was</td><td>The cold dog was</td><td>tast, circular, price</td><td>3</td><td>0.85</td></tr><tr><td>The hot dog was</td><td>the cold dog was</td><td>sold, more, circular</td><td>12</td><td>0.64</td></tr><tr><td>... the hot dog was</td><td>... the cold dog was</td><td>flavor, tasted, addictive</td><td>0</td><td>1.00</td></tr></table>

Table 2: The same edibility contrast (hot vs cold), read at the final token (underlined) under three frames: both prompts capitalised (matched), the second prompt’s first token lowercased (a mismatch), and both behind a shared preamble $( \dots { } . \mathrm { = } ^ { \prime \prime } \mathrm { M y }$ grandmother said that”). Readout is the verbatim top-3 at L20; food is the best food-token rank there; cos is the cosine of the L20 difference direction against the preambled one. The mismatch pushes food down the ranking and rotates the read $( 0 . 8 5  0 . 6 4 )$ ; the preamble reads food most stably. The mismatch degrades only the mid-stack read: by L24 all three recover to clean food.

Why does the mismatch hurt? The capitalization difference on its own is large and illegible. The pure “The” versus “the” contrast (“The hot dog was” versus “the hot dog was,” same words otherwise) has a difference vector of norm about 40 at L20, and reads as junk (“NEY, ENE”). That junk is a large fraction of the food contrast itself (norm about 68 at L20), so a start mismatch rotates the reading toward it. The cause is position, not capitalisation: any difference on the first token would do the same. The rule follows. Match the prompts everywhere except the target, and keep the target several tokens past the start with a shared lead-in. For a strong lexical contrast the preamble is only a refinement: the food signal still recovers by the late layers, and only the mid-stack read is muddied.

Match the current token. The token at the read position is processed heavily by the early layers. An incidental difference there, even a nearly meaningless one, adds a large surface component that swamps the difference of interest. $\mathrm { ~ A ~ } 2 \times 2$ makes this precise (Table 3): cross a context (reading a novel versus watching a lecture) with a synonym at the final word (boring versus dull), and read the same novel-versus-lecture difference. When the final word matches, both poles read the context cleanly from the middle layers (Table 4): the novel pole reads “novels, literary, readers,” the lecture pole reads “lectures, seminar, students, videos.” When the final word is a synonym mismatch, the difference also carries the boring-versus-dull swap. Its norm is about ten times larger at L1 (30 versus 3–5), and the swap’s suffix morphology (“ly, ed, ers, ing”) dominates the readout (Table 5). The context is buried, and surfaces only at the final layers (L30–32, “readers” on the novel pole, “video, youtube” on the lecture pole), even there mixed with fragments. Matching the current token clears the readout. This is why the trajectories here are read at shared-token positions such as “dog” and “was.” The same rule applies to baseline subtraction: snippets chosen to end in the target’s final token cancel the current-token component.

<table><tr><td></td><td>Context Prompt</td></tr><tr><td>novel</td><td>I spent the whole afternoon reading the novel and found it extremely [boring / dull]</td></tr><tr><td>lecture</td><td>I spent the whole afternoon watching the lecture and found it extremely [boring / dull]</td></tr></table>

Table 3: Prompts for the current-token 2 × 2. The read contrasts the two contexts (novel versus lecture); the final word (boring versus dull) is crossed with them, matched or mismatched, and read at the final token (underlined).

<table><tr><td>L</td><td>+ reading the novel</td><td>— watching the lecture</td></tr><tr><td>4</td><td>novels, literary, rette</td><td>participants, seminar, attendees</td></tr><tr><td>20</td><td>literary, satir, novels</td><td>lectures, presenter, students</td></tr><tr><td>28</td><td>readers, Readers, reader</td><td>lectures, lecture, videos</td></tr><tr><td>32</td><td>readers, reader, literary</td><td>lectures, lecture, teaching</td></tr></table>

Table 4: Matched current token (both prompts end in “dull”). The novel-versus-lecture difference reads both poles cleanly: the novel/reading pole (+) and the lecture/watching pole (−). Verbatim top-3 (“rette” is a fragment).

<table><tr><td>L</td><td>|||</td><td>+ reading the novel</td><td>– watching the lecture</td></tr><tr><td>4</td><td>34.5</td><td>ed, ers, iveness</td><td>ly, cium, imi</td></tr><tr><td>20</td><td>47.5</td><td>ers, er, ively</td><td>iosis, clip, cone</td></tr><tr><td>28</td><td>64.5</td><td>ers, er, ership</td><td>but, I, video</td></tr><tr><td>30</td><td>76.2</td><td>ers, er, readers</td><td>video, clip, Video</td></tr><tr><td>32</td><td>26.9</td><td>er, ers, readers</td><td>Videos, youtube, video</td></tr></table>

Table 5: Mismatched current token (novel/dull minus lecture/boring). The boring-versus-dull suffix morphology (ly, ed, ers, ing) dominates, and the difference norm is many times the matched pair’s. The context surfaces only at L30–32 (“readers” on the novel pole; “video, youtube” on the lecture pole), still mixed with fragments. Verbatim top-3.

Match the predicted next token. The two rules above concern the read position’s input. A third concerns its output. The logit lens at a position reads mostly that position’s own next-token prediction. This can bury content the position holds for later use, since downstream positions read a residual through attention. To surface that content, choose a contrast whose two prompts predict the same next token at the read position. The shared prediction then cancels, and the difference keeps only what differs downstream. Read at “Japan” in “The capital of Japan” versus “The currency of Japan” (Table 6), both prompts predict “is.” The difference splits by relation at the late layers (Table 7): capital−currency reads place vocabulary, currency−capital reads monetary vocabulary. France behaves the same on the currency side (“denomination, exchange, devalue, pegged”), with a noisier capital side (“city, location”). Now read the same contrast one token later, at the “is” of “The capital of Japan is” versus “The currency of Japan is.” Here the two prompts predict different next tokens, and the difference reads those predictions instead: capital−currency reads capital cities (“Tokyo, London, Paris, Beijing”), currency−capital reads currency names (“yen, dollar, yuan”). Matching the next token reads how a position applies a relation. Reading where the next tokens differ reads the answer that relation produces.

<table><tr><td>Capital-of</td><td>Currency-of</td></tr><tr><td>The capital of Japan</td><td>The currency of Japan</td></tr><tr><td>The capital of Japan is</td><td>The currency of Japan is</td></tr></table>

Table 6: Prompts for the next-token contrast, read positions underlined. The first pair is read at the “Japan” position (shared next token “is”); the second at the final “is” (differing next tokens).
<table><tr><td>L</td><td>Raw (both)</td><td>capital – currency</td><td>currency – capital</td></tr><tr><td>24</td><td> $\mathrm { i s } , \prime \mathrm { s }$ </td><td>loc, Location, locate</td><td>currencies, denomination, exchanges</td></tr><tr><td>28</td><td> $\mathrm { i s } , , \mathrm { a n d }$ </td><td>lat, obe, location</td><td>circulated, denomination, predec</td></tr><tr><td>32</td><td> $\mathrm { i s } , \mathrm { , } \mathrm { \quad } \mathrm { w a s }$ </td><td>capitals, city, Population</td><td>pegged, denomination, devalue</td></tr></table>

Table 7: Matching the next token isolates a stored relation. Read at the “Japan” position of “The capital of Japan” versus “The currency of Japan” (Phi-2). Both prompts predict “is,” so the raw read is the same for each and cancels in the difference; the difference then splits by relation, place vocabulary on the capital side and monetary vocabulary on the currency side. Tokens are verbatim top-3 (fragments such as $^ { \prime \prime } \mathrm { l o c } , ^ { \prime \prime } ^ { \prime \prime } \mathrm { l a t ^ { \prime \prime } }$ are location sub-words; “predec,” “obe” are noise).

## 2.6 Baseline subtraction: the single-prompt variant

A contrastive projection needs a second prompt. One generic alternative uses none. From the target’s residual, subtract the mean residual of many text snippets of the same length, read at the same position. The same length matters for a positional reason: it keeps the read at the same index in every snippet, so position-dependent structure (strong under rotary embeddings) is shared and cancels in the average, rather than leaking into the readout. The subtraction then removes what is common to text of that kind and leaves what is specific to the target. The baseline should match the target’s style. A mean over English prose sentences cancels the function-word and sentence-frame content of a prose target cleanly. A mean over arbitrary web text (code, lists, markup) leaves a noisier remainder. The closer the baseline’s style to the target, the cleaner the readout. Multi-contrast triangulation (§2.7) is the limit of this: a baseline built from the target’s own frame.

Baseline subtraction is coarser than a matched pair. A matched pair cancels the shared computation almost exactly. The prose mean cancels only the generic style, so the target’s content clears the readout only at the late layers, where its prediction is strong. Reading the average itself shows what is removed (Table 9). For “My grandmother said that the hot dog was” (Table 8), the mean of 100 English prose sentences reads generic sentence-final content (articles and punctuation). The difference surfaces the target’s food vocabulary.
<table><tr><td>Role</td><td>Text</td></tr><tr><td>Target</td><td>My grandmother said that the hot dog was</td></tr><tr><td>Baseline</td><td>mean residual of 100 English prose sentences of the same length</td></tr></table>

Table 8: Prompts for baseline subtraction: one target, and the prose mean it is read against. Both read at the final token (underlined).

<table><tr><td>L</td><td>Raw (target)</td><td>Prose average</td><td>Baseline-subtracted</td></tr><tr><td>20</td><td>not, more, made, a</td><td>in, first, the</td><td>not, made, always, too</td></tr><tr><td>24</td><td>not, more, a, better</td><td>the, in, {,}</td><td>delicious, better, easier, more</td></tr><tr><td>28</td><td>more, better, too, not</td><td>the, a, {,}</td><td>delicious, cooked, hotter, tast</td></tr></table>

Table 9: Baseline subtraction on “My grandmother said that the hot dog was” (Phi-2, final position), against the mean of 100 English prose sentences of the same length. The prose average reads generic sentence-final content (articles, punctuation); the difference surfaces the target’s food vocabulary, which clears the readout at the late layers (L28: “delicious, cooked, hotter, tast”).

## 2.7 Multi-contrast triangulation

A single-pair difference spans every axis on which the two prompts differ, not only the intended one. Multi-contrast triangulation isolates the intended axis: contrast the target against several baselines that share that axis but differ on the incidental ones, then average the difference vectors. The incidental axes vary across baselines and cancel; the axis common to every contrast survives. To isolate edibility in “hot dog,” for instance, read it against angry, small, and pet dog, all non-edible: edibility is the one difference every contrast shares, so anger, size, and petness average away. It is the middle ground between a single pair and subtracting a generic corpus average (§2.6): matched baselines cancel the incidental axes without also cancelling the sentence frame.

A companion paper explains why averaging surfaces the shared content (Tuomi, 2026). A tokenaligned component surfaces in the top-K only above a visibility threshold $f ^ { * } = 2 \ln ( 2 V ) / ( d +$ $2 \ln ( 2 V ) ) \approx 2 \ln ( 2 V ) / d$ (the approximation holds because d ≫ ln V). Averaging N matched difference vectors suppresses the incoherent remainder by ${ \sqrt { N } } ,$ which lowers that bar. Averaging does not sharpen the signal. It removes the unaligned remainder. Entity-identity contrasts (a name, a city) differ on little besides the target and read cleanly from a single pair; axes that co-vary with several others benefit most. The result is read at the granularity the contrast design provides, so a surfaced token may name a bundle of co-activated features rather than a single one. We exercise triangulation where it earns its keep: in §3.2 it recovers the noun-internal food read in models where a single pair leaves it below the readout’s detection bar.

## 2.8 Using a contrast: decompose and test

A two-item contrast rarely lands on a single axis. When a readout looks mixed, we take it apart: name the concept axes the pair might span, build each as its own data-derived contrast (a mean over one noun set minus a mean over another at the read position), and read each through $W _ { U }$ to confirm it names its concept. Projecting the original difference onto the set shows which axes it carries. We then test each for causality by injecting it alone, moving one state toward the other along that axis and measuring the effect on the prediction, rather than trusting the raw readout. This is also why triangulation helps: averaging over baselines that share the intended axis cancels the others. Reading a category as a region of a space of interpretable quality dimensions is the conceptual-spaces view of concepts (Gärdenfors, 2000), in the tradition of the semantic differential (Osgood et al., 1957). We work the method through on the hot-dog contrast in §3.1.

## 3 Lexical disambiguation

## 3.1 Compound noun: hot dog

<table><tr><td></td><td>Prompt</td><td>Top next-token predictions</td><td>Greedy continuation</td></tr><tr><td>Hot dog (food)</td><td>The hot dog was</td><td>too (0.088), more (0.080), cooked (0.063)</td><td>too spicy for the child</td></tr><tr><td></td><td>Cold dog (animal) The cold dog was</td><td>sh[ivering] (0.447), shaking (0.036)</td><td>shivering in the snow</td></tr></table>

Table 10: The two prompts of the compound-noun contrast, read at “dog” and “was” (underlined). The top next-token tokens are ambiguous (“too, more”); the greedy continuation makes the food-versus-animal reading clear.

What the distinction decomposes into. The contrast is meant to isolate edibility, but its poles are different concepts: the hot pole reads food (“tast, delicious, flavor”), the cold pole reads animal care (“grooming, Paw, breeds”). Decomposing it (§2.8), we build three candidate axes at the read position, edibility, petness, and temperature (Table 11), and project the hot-cold difference onto them. It carries all three (at L20, edibility +21, petness −20, temperature +15): the compound adds edibility and suppresses the petness that “dog” carries, and the prompts differ in temperature besides.

Which axis drives the food-versus-animal prediction? We inject each in turn, moving the cold-dog state toward hot-dog along one axis and measuring how far the next token moves from animal toward food (Table 12). Edibility alone barely helps (0.03 at L20), petness alone barely helps (0.01); neither switch works by itself. Both together recover almost all of the effect (0.98, against 1.00 for the full difference), far more than the sum. The food/animal boundary is diagonal in this plane:

<table><tr><td>Axis</td><td>Built as (mean – mean, frame &quot;The X was&quot;)</td><td>Reads (L20)</td></tr><tr><td>Edibility (E)</td><td>edible nouns (steak, sausage, burger, . . .) – objects (rock, chair, brick, . . .)</td><td>delicious, cooked, tasted</td></tr><tr><td>Petness (P)</td><td>pet animals (puppy, kitten, poodle, ...) – the same objects</td><td>roaming, adorable, wandering</td></tr><tr><td>Temperature (T)</td><td>&quot;hot X&quot; — &quot;cold  $X ^ { \prime \prime }$  water, ...)</td><td>over neutral nouns (soup, coffee, steam, boiling, hotter</td></tr></table>

Table 11: The three concept axes, each built as a multi-contrast: a mean over one noun set minus a mean over another, at the read position. Reading each direction through $W _ { U }$ confirms it names its concept. Averaging over many nouns cancels the idiosyncrasies of any single one.

food needs high edibility and low petness at once, not one or the other. Temperature is inert, doing nothing alone (0.00) and adding nothing to the pair. So the contrast spans three axes but only two are causal, and they act as a plane; this makes semantic sense, since a hot dog can be cold or warm without affecting its hotdogness. The axes are correlated (cosine 0.37), so the split is not perfectly clean, they are data-derived from particular noun sets, and this is one prompt pair in one model.

<table><tr><td rowspan="2">L</td><td colspan="2">projection</td><td colspan="5">food fraction, injected</td></tr><tr><td>E</td><td>P</td><td>T E</td><td>P</td><td>T</td><td>E+P</td><td>E+P+T</td></tr><tr><td>12</td><td>+15</td><td>-11 +17</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.24</td><td>0.21</td></tr><tr><td>16</td><td>+16</td><td>-14 +16</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.55</td><td>0.48</td></tr><tr><td>20</td><td>+21</td><td>-20 +15</td><td>0.03</td><td>0.01</td><td>0.00</td><td>0.98</td><td>0.99</td></tr><tr><td>24</td><td>+43</td><td>-37 +18</td><td>0.53</td><td>0.01</td><td>0.00</td><td>1.00</td><td>1.00</td></tr></table>

Table 12: Decomposing “hot dog” − “cold dog” into concept axes: edibility (E), petness (P), temperature (T). Projection is the component of the difference on each axis. Foodfraction is P(food)/(P(food) + P(animal)) at the next token after moving the cold-dog run toward the hot-dog run along that axis (E makes it more edible, P makes it less pet) at the read position (baseline cold 0.00, hot 1.00; the full difference recovers 1.00). Neither E nor P alone crosses the boundary, but the E+P plane recovers it; temperature is inert (E+P+T equals E+P). Edibility alone gains potency only at the readout (L24).

Per-position trace. We read at each token (Table 13). “dog” is the same token in both prompts, so the L0 contrast is zero. Both positions first read temperature (“hot, molten” at L1). The food pole then separates: at “dog” it reads “fried” by L5, and at the “was” prediction site it builds to “tast, charred” by L28. The animal pole is the mirror image, legible only later and clearest at the readout (L28: “whine, Paw” at “dog,” “grooming, shudder” at “was”). Both poles crystallize at the prediction site as the food and animal continuations of the two prompts. How the food signal reaches “was,” an MLP write at “dog” routed forward by attention, is decomposed next.

Sub-layer decomposition at L4–L5 confirms the MLP recognizes the compound at L4, not attention. The MLP contrastive norm at L4 (20.0) dominates the attention norm (3.8). “fried” appears in the MLP output but not the attention output. L5 attention writes orthogonally to the food direction (cos < 0.08 across all contrasts tested).

Attention routing at L5, position “was”: Head 19 attends to the “dog” position with weight 0.911 in the hot-dog context versus 0.735 in cold-dog. It reads the compound-noun information from “dog” and writes it to “was” (its write is read out below).

<table><tr><td></td><td colspan="2"> $\mathrm { A t } ^ { \prime \prime } \mathrm { d o g ^ { \prime \prime } }$ </td><td colspan="2"> $\mathrm { A t } \ ^ { \prime \prime } \mathrm { w a s } ^ { \prime \prime }$ </td></tr><tr><td>L</td><td>food (+)</td><td>animal (−)</td><td>food (+)</td><td>animal (−)</td></tr><tr><td>1</td><td>hot, molten</td><td></td><td>hot, fiery</td><td></td></tr><tr><td>5</td><td>fried</td><td>Breed</td><td>boiled</td><td></td></tr><tr><td>20</td><td>flavor</td><td></td><td>tast</td><td>euth, Paw</td></tr><tr><td>28</td><td>vendor, stand</td><td>whine, Paw</td><td>tast, charred</td><td>grooming, shudder</td></tr></table>

Table 13: Per-position contrastive readout for hot dog versus cold dog, both poles: food (hot dog, +) and animal (cold dog, −), cumulative read at each position (Phi-2). Both poles start as temperature and split into food and animal by the prediction site: the food pole is legible at “dog” by L5, the animal pole only at the readout (L28). “—” marks a pole with no legible read yet.

Multi-contrast convergence. The food-compound direction is stable across reference points. Five contrasts (hot dog minus cold/angry/old/pet/stray dog) give pairwise cosine 0.72–0.84 at the dog position. The signal is about food-compound identity, not temperature or emotion.

The detection bar. The visibility threshold f<sup>∗</sup> (Tuomi, 2026) makes the soup-versus-signal call quantitative: a token clears the isotropic bath (the alignments an unaligned, random component would produce by chance) only if its energy fraction $f = \cos ^ { 2 } ( x , u _ { t } )$ exceeds $f ^ { * } = 2 \ln ( 2 V ) / ( d +$ 2 ln(2V)), which is 0.89% for Phi-2. At the prediction site, food clears the bar where the trajectory reads cleanly (“delicious” at f = 1.9%, 2.1× the bar, by L24) and falls below it in the middle layers, where the readout is fragmentary (f < 0.7%, L10–16). At the noun the food token is weaker: “fried” tops the raw logit lens at L5 but its energy fraction is only 0.6%, below the bar. The logit lens ranks by an unnormalised dot product, which a token’s unembedding norm inflates independently of alignment, so “fried” can head the list at the noun without the difference vector being strongly aligned with it. It heads the printed readout, but its alignment is below f , so it does not stand out from the isotropic bath: a top rank in the logit lens is not the same as clearing the bar, and the noun read is weak rather than robust. What clears the bar is the emission at the prediction site, not the read at the noun. The trained probe (§3.2) confirms the distinction is present regardless, because it does not read through this bar.

Neuron and head detail. The MLP write at “dog” is distributed. The ten neurons pushing hardest along the food direction carry only ∼10% of it. The rest spreads across hundreds. Some write food (one writes “cooked, foods, dishes”), some suppress animal content (another writes “dogs, puppies”). The routing head L5.H19, flagged above by its attention to “dog,” writes “cooked, eaten” into “was.” So the food direction at the prediction position is written by a distributed MLP population and moved forward by one head, not by any single neuron. The read points to where this happens. Patching (below) confirms the path is causal.

## Activation patching at multiple layers and positions (Table 14):

The patching traces the two-hop chain. (1) Patching “dog” early destroys the food reading (L2: “panting heavily”), because the compound meaning lives there; by L20 the patch has no effect (“too spicy”), the compound having already been routed forward. (2) Patching “hot” never matters (“too hot to eat”): L0 attention has already copied it to “dog.” (3) Patching “was” is the mirror image, harmless early (L2: “the hamburger”) and removing food from L12 on until it is gone late (L20: “shivering in the cold”). This matches attention copying the compound from “dog” to “was” in between.

<table><tr><td>Patched</td><td>Layer</td><td>P(food)</td><td>Greedy continuation</td></tr><tr><td>(baseline)</td><td></td><td>0.16</td><td>too spicy for the child</td></tr><tr><td>dog</td><td>2</td><td>0.00</td><td>panting heavily, so the</td></tr><tr><td>dog</td><td>8</td><td>0.02</td><td>more appealing to the children</td></tr><tr><td>dog</td><td>20</td><td>0.13</td><td>too spicy for the little girl</td></tr><tr><td>hot</td><td>2</td><td>0.13</td><td>too hot to eat, so</td></tr><tr><td>was</td><td>2</td><td>0.16</td><td>more expensive than the hamburger</td></tr><tr><td>was</td><td>12</td><td>0.06 0.00</td><td>placed in the oven to cook</td></tr><tr><td>was</td><td>20</td><td></td><td>shivering in the cold weather</td></tr></table>

Table 14: Activation patching of the hot-dog prompt: the residual at one position and layer is replaced with the cold-dog value, then generation continues greedily (Phi-2). P(food) is the food mass at the next token. Patching “dog” early destroys the food reading (“panting”), and by L20 has no effect, the compound having moved to “was”; patching “was” is the mirror, harmless early (“the hamburger”) and destroying food late (“shivering”); patching “hot” never matters.

Mechanism. Attention at L0 copies “hot” to “dog.” The MLP at L4 recognizes the compound at “dog” (“fried” first appears in the MLP output). Attention at L5 broadcasts it from “dog” to “was” (H19, attn=0.91). The contrastive projection flagged each stage. Activation patching at three positions and multiple layers confirms the flow. The distinction traced here is the edibility-petness plane established at the top of this section: the trace and patching locate where that plane is written and how it reaches the prediction site.

## 3.2 Cross-model: what generalizes and what is model-specific

We re-run the compound-noun decomposition on three architectures chosen to differ where it should matter. Phi-2 has dense multi-head attention, a GELU MLP, and a parallel residual. Pythia-1.4B (GPT-NeoX) has dense attention, GELU, and a parallel residual. Qwen2.5-1.5B has groupedquery attention, SwiGLU, and a sequential residual. The residual decomposition mlp[L] = h[L + 1] − h[L] − attn[L] (with attn[L] at the output projection) is valid for both parallel and sequential streams, so the attention-vs-MLP split is measured identically in all three. We read the compound noun prompts (Table 10) at the noun and prediction positions inside a shared preamble (§2.5). The bare prompt is not a fair instrument here: without the preamble Qwen2.5’s prediction readout decodes to product/commerce tokens (“priced, sold, artisan”) that resolve to food vocabulary once the preamble stabilizes the read.

Two results follow, one robust and one architecture-specific (Tables 17 and 18). Robust. In all three models food content reaches the prediction position through later attention and reads cleanly there (Table 18): from L5 in Phi-2, L14 in Pythia, and L12 in Qwen (with multilingual flavor tokens, as

expected for a multilingual model). This resolve-at-the-noun, route-to-the-prediction shape holds across dense and GQA attention, GELU and SwiGLU MLPs, and parallel and sequential residuals. The greedy continuations confirm the distinction is behavioural, not an artefact of the readout (Table 15): Phi-2 and Qwen continue the hot dog as food and the cold dog as an animal, while the smaller Pythia decodes degenerately here.
<table><tr><td>Model</td><td>&quot;The hot dog was&quot; ...</td><td>&quot;The cold dog was&quot; ...</td></tr><tr><td>Phi-2 (2.7B)</td><td>too spicy for the child</td><td>shivering in the snow</td></tr><tr><td>Pythia-1.4B</td><td>a hit ...</td><td>still there, and still there ..</td></tr><tr><td>Qwen2.5-1.5B</td><td>a popular food item</td><td>barking at the moon</td></tr></table>

Table 15: Greedy continuations of the two prompts. Phi-2 and Qwen continue the hot dog as food and the cold dog as an animal (Phi-2 matches the continuations used elsewhere). The smaller Pythia-1.4B decodes degenerately here, repeating itself, so its food reading shows better in the readout (Table 18).

The noun-internal read needs triangulation. A single hot/cold pair reads food at the noun in Phi-2, where “fried” enters the top ranks at L4, the layer the MLP writes it. It does not in Pythia or Qwen, where food never reaches the top-10. This is an instrument limit, not an absence. We average the food-compound direction over five baselines (hot dog minus {cold, angry, old, pet, stray} dog), the triangulation of §2.7. This recovers a legible noun read in Pythia (best food rank 3 against 17 for the single pair, at L15–17) and lifts Qwen from rank 60 to 20, food-service vocabulary that stops short of the top-10 (Table 18). So the compound has a token face at the noun in Phi-2 and Pythia: read directly in one, recovered by triangulation in the other. In Qwen it surfaces only partially. We do not claim every intermediate site carries a clean token face, since Qwen’s stays partial even after triangulation. But the single-pair illegibility is mostly an instrument effect, not evidence that the channel has no token form. Triangulation raises the energy fraction f where a component sits near the readout’s detection bar f<sup>∗</sup> (Tuomi, 2026): averaging the five baselines lifts the Pythia noun read to about 1.3 × f<sup>∗</sup>, while Qwen’s stays below f<sup>∗</sup> even then, matching its partial recovery.

A probe confirms the distinction is present, token face or not. A trained probe settles whether the illegible noun read is an absence or only an unemitted computation. We train a linear probe to separate clear food nouns (steak, burger, bacon, . . . ) from animal nouns (poodle, kitten, rabbit, . . . ) at the noun position, then apply it to the “dog” of “hot dog” and “cold dog” (Table 16). In all three models it reads hot dog’s noun as food and cold dog’s as animal with near-certainty (P(food) of 1.00, 0.99, 1.00 for hot dog and near zero for cold dog, at five-fold probe accuracy 1.00), including Qwen, where the contrastive read surfaces no clean food token at the noun. The compound distinction is computed in every model; the token face is what varies. This is the trade-off with probing in miniature: the probe is the more sensitive detector, but it needs the food-versus-animal labels, whereas the contrastive read needs only the prompt.

<table><tr><td>Model</td><td>hot dog P(food)</td><td>cold dog P(food)</td><td>noun read, single pair</td></tr><tr><td>Phi-2 (2.7B)</td><td>1.00</td><td>0.00</td><td>fried (rank 0)</td></tr><tr><td>Pythia-1.4B</td><td>0.99</td><td>0.01</td><td>— (rank 17)</td></tr><tr><td>Qwen2.5-1.5B</td><td>1.00</td><td>0.00</td><td>— (rank 60)</td></tr></table>

Table 16: The compound distinction is decodable even where no token face surfaces. A linear probe trained on clear food versus animal nouns (five-fold accuracy 1.00 in each model) classifies the “dog” of “hot dog” as food and of “cold dog” as animal in all three models. The contrastive single-pair read at the noun is legible only in Phi-2; the computation is present regardless.

<table><tr><td></td><td>Phi-2</td><td>Pythia-1.4B</td><td>Qwen2.5-1.5B</td></tr><tr><td>attention</td><td>dense MHA</td><td>dense MHA</td><td>GQA</td></tr><tr><td>MLP / residual</td><td>GELU / par.</td><td>GELU / par.</td><td>SwiGLU / seq.</td></tr><tr><td>food reaches prediction site</td><td>L5</td><td>L14</td><td>L12</td></tr></table>

Table 17: Architecture across three models, and the layer at which food reaches the prediction site through later attention.

<table><tr><td>Model</td><td>Prediction-site read</td><td>Noun, single pair</td><td>Noun, triangulated</td></tr><tr><td>Phi-2 (2.7B)</td><td>delicious, charred</td><td>fried (rank 0)</td><td>fried (rank 1)</td></tr><tr><td>Pythia-1.4B</td><td>burgers, cuisine, delicious</td><td>— (rank 17)</td><td>sandwiches, eaten, lunch (rank 3)</td></tr><tr><td>Qwen2.5-1.5B</td><td>flavor, flavorful, tastes</td><td>— (rank 60)</td><td>toppings, grill, taste (rank 20)</td></tr></table>

Table 18: The compound-noun food read across three models. At the prediction site the method reads food in all three. At the noun a single pair reads food only in Phi-2; triangulation recovers it in Pythia (rank 3) and lifts Qwen partway (rank 20). “—” = no legible food token in the single-pair read; best food rank in parentheses.

## 3.3 Other disambiguation cases

The same method reads disambiguation driven by a verb sense or a particle, the two poles giving the two senses (Table 19).
<table><tr><td>Contrast</td><td>Pole A reads</td><td>Pole B reads</td></tr><tr><td>&quot;He caught a cold&quot; / &quot;... fish&quot; (verb sense) &quot;He was fired up&quot; / “... fired&quot; (particle)</td><td>fever, coughing, cough, flu excited, ready, energetic</td><td>proudly, reel, bait, trout sued, blacklist, lawsuits</td></tr></table>

Table 19: Verb-sense and particle disambiguation read at the final position (L28). Each pole reads its sense: illness versus fishing, enthusiasm versus dismissal.

A noun can also be disambiguated by a modifier. Contrasting “The steep bank was” against “The closed bank was,” read at the shared word “bank” (the current-token rule of §2.5, since the modifiers differ), the two poles read the two senses of the noun (Table 20): the steep pole reads the riverbank sense (“slopes, erosion, steep”), the closed pole the financial sense (“doors, branch, closed”).

<table><tr><td>L</td><td>steep pole (riverbank)</td><td>closed pole (financial)</td></tr><tr><td>20</td><td>slopes, slope, surrounding</td><td>Frankfurt, entities</td></tr><tr><td>24</td><td>slopes, erosion, steep</td><td>doors, closed, Doors</td></tr><tr><td>28</td><td>slopes, slope, erosion</td><td>branch, window</td></tr></table>

Table 20: A noun disambiguated by a modifier: “The steep bank was” versus “The closed bank was,” read at the shared “bank” (Phi-2). The steep pole reads the riverbank sense, the closed pole the financial sense.

## 4 Metaphor: domain routing, not a figurativity flag

Metaphor is a clean test of what the readout can see. Does the model mark figurative language with a single “figurativity” feature, or with something more local? Four adjectives (cold, sharp, bright, heavy) are each used once literally and once metaphorically, in a shared frame that ends just before the two senses diverge (Table 21).

<table><tr><td>Word</td><td>Literal prompt</td><td>Metaphorical prompt</td></tr><tr><td>cold</td><td>The ice in the bucket was extremely cold. The temperature was</td><td>The reception at the party was extremely cold. The atmosphere was</td></tr><tr><td>sharp</td><td>The razor was extremely sharp. The blade was</td><td>The review was extremely sharp. The tone was</td></tr><tr><td>bright</td><td>The spotlight was extremely bright. The light was</td><td>The class was extremely bright. The child was</td></tr><tr><td>heavy</td><td>The barbell was extremely heavy. The weight was</td><td>The room was extremely heavy. The mood was</td></tr></table>

Table 21: The four literal/metaphorical prompt pairs, each read at the final token (“The [role] was,” underlined).

Each prompt is built to end at “The [role] was,” and we read at that final token, a few tokens past the adjective. The predicted next token is meant to differ: “The temperature was” predicts a

temperature, “The atmosphere was” predicts a mood. Each pole then reveals the domain it routes to. At L24 the literal pole returns the physical domain and the metaphorical pole a specific target domain, different for each word (Table 22).
<table><tr><td>Word</td><td>Literal pole</td><td>Metaphorical pole</td></tr><tr><td>cold sharp bright heavy</td><td>higher, Celsius, Fahrenheit blade, blades, stainless blinding, intensity, harsh load, exert, Hercules</td><td>tense, atmosphere, mood, vibe tone, sarcastic, condescending proud, gifted, amazed, grades mood, tense, gloomy, bleak</td></tr></table>

Table 22: Contrastive readout at L24 for the four literal/metaphorical contrasts. Each metaphorical pole routes to its own target domain: cold and heavy to emotion, sharp to social tone, bright to intelligence.

The routing direction is stable across sentences for each word (pairwise cosine 0.64–0.88 over four pairs). To check it is not memorised from the particular prompts, we extract it from three pairs and inject it into the held-out fourth (leave-one-out). Table 23 shows the four folds for cold. Injecting toward emotion on a held-out literal prompt surfaces “tense, chilly.” Injecting toward temperature on a held-out metaphorical prompt surfaces “below, freezing.” This holds in every fold. The effect is graded, and generating a few tokens after the injection shows it (Table 24). At unit magnitude only cold reverses to its literal domain (“below freezing”); doubling the injection reaches the literal domain for sharp (“sharp”) and bright (“blinding”) as well. Heavy is the exception, staying at “too dark,”. So the routing direction is domain-specific for all four (the cosine matrix below); the magnitude needed to flip a whole continuation just varies by word.

<table><tr><td>Held-out fold</td><td>Literal prompt, —dir (toward emotion)</td><td>Metaphorical prompt, +dir (toward temperature)</td></tr><tr><td>0</td><td>tense, chilly, freezing</td><td>so, below, freezing</td></tr><tr><td>1</td><td>tense, icy, so</td><td>so, below, very</td></tr><tr><td>2</td><td>chilly, tense, unw</td><td>so, below, much</td></tr><tr><td>3</td><td>so, chilly, tense</td><td>very, below, freezing</td></tr></table>

Table 23: Leave-one-out for cold. The routing direction is built from three literal/metaphorical pairs and injected (±1×) into the held-out fourth. Top-3 readout tokens after injection; the direction shifts a heldout literal prompt toward emotion and a held-out metaphorical prompt toward temperature, so it is not memorised from the extraction prompts.
<table><tr><td>Word</td><td>Literal direction +1 ×</td><td>+2×</td></tr><tr><td>cold</td><td>below freezing, and everyone</td><td>below freezing</td></tr><tr><td>sharp</td><td>very negative</td><td>sharp</td></tr><tr><td></td><td>bright too young to understand</td><td>blinding</td></tr><tr><td></td><td>heavy too dark for the party</td><td>too dark for the party</td></tr></table>

Table 24: Greedy continuations of each word’s metaphorical prompt with the literal routing direction injected at 1× and 2× (Phi-2, L24). At 1× only cold reverses to its literal domain (“below freezing”); at 2× sharp (“sharp”) and bright (“blinding”) reach it too. Heavy stays at “too dark,” a mood word, not its literal weigh domain.

Injection is bidirectional and graded. Adding the cold direction to “The reception was extremely cold. The atmosphere was” shifts “chilly” (social) toward “below” (temperature), with the top-1 flip at 0.65× its natural magnitude. Subtracting it from the literal prompt shifts “below” toward “tense” (emotion) by 1.5×.

The decisive test is cross-domain. If the model had one literal/figurative axis, a word’s routing direction should move any context toward that context’s own literal sense. It does not (Table 25). Injected into a sharp tone or a heavy mood, the cold direction drives the continuation to temperature (“below zero”), not toward sharpness or weight. The sharp direction does the reverse, driving a cold atmosphere or a heavy mood to “sharp.” Each direction imposes its own target domain whatever the context, so the mapping is source-specific, not a generic figurativity flag.
<table><tr><td>Direction → context</td><td>Baseline</td><td>Injected (3×)</td></tr><tr><td>cold → &quot;The tone was&quot; (sharp)</td><td>very negative</td><td>below zero</td></tr><tr><td>cold → &quot;The mood was&quot; (heavy)</td><td>somber</td><td>below zero</td></tr><tr><td>sharp → &quot;The atmosphere was&quot; (cold)</td><td>chilly, unwelcoming</td><td>sharp and unwelcoming</td></tr><tr><td>sharp → &quot;The mood was&quot; (heavy)</td><td>somber</td><td>sharp</td></tr></table>

Table 25: Cross-domain injection at 3× (Phi-2, L24): one word’s routing direction injected into another word’s context, greedy continuation. Each direction imposes its own target domain regardless of the context. The cold direction drives a sharp tone or a heavy mood to temperature (“below zero”), the sharp direction drives a cold atmosphere or a heavy mood to sharpness (“sharp”). A single literal/figurative axis would instead move each context toward its own literal sense.

The cross-domain cosines agree (Table 26). Cold and heavy both map a physical scale onto emotion and share structure (0.58). Bright maps onto intelligence and is nearly orthogonal to the rest (0.03–0.14).
<table><tr><td></td><td>cold</td><td>sharp</td><td>bright</td><td>heavy</td></tr><tr><td>cold</td><td>1.00</td><td>0.28</td><td>0.06</td><td>0.58</td></tr><tr><td>sharp</td><td>0.28</td><td>1.00</td><td>0.03</td><td>0.37</td></tr><tr><td>bright</td><td>0.06</td><td>0.03</td><td>1.00</td><td>0.14</td></tr><tr><td>heavy</td><td>0.58</td><td>0.37</td><td>0.14</td><td>1.00</td></tr></table>

Table 26: Pairwise cosine between the four routing directions at L24. Directions that share a target domain (cold, heavy) correlate; bright, which routes to a different target, is orthogonal to all.

We read this as a claim about representation, scoped to these four words. We find no single figurativity feature. What we call metaphor here is a set of domain-to-domain mappings, and the projection reads one mapping at a time. The direction that separates a literal from a figurative use is fixed by the source and target domains it connects, not by figurativity itself. This is why a single literal/metaphorical axis is inconsistent across words, and why the reading is legible only when the contrast holds the domain pair fixed.

## 5 Recall versus hallucination

When the model hallucinates, it produces a confident but fabricated answer for a fictional entity. The contrastive projection shows what its retrieval surfaces in token space: specific facts for real entities, and nothing beyond name fragments for fictional ones.

Design: We construct matched pairs where one prompt elicits genuine recall and the other elicits hallucination, keeping the frame identical (Table 27):
<table><tr><td>Prompt</td><td></td><td>Prediction</td></tr><tr><td>Real Fictional</td><td>Nikola Tesla, born in 1856, invented the Ludvig von Vogelkirche, born in 1859, invented the</td><td>Tesla coil... first practical electric motor...</td></tr><tr><td>Real Fictional</td><td>Marie Curie, born in 1867, discovered Helena Brandström, born in 1871, discovered</td><td>the elements polonium and radium.. a new species of moth. . .</td></tr></table>

Table 27: Matched real and fictional entity prompts, read at the final token (underlined). Both elicit confident, specific predictions.

Both sides produce confident, specific answers. But the contrastive projection at L28 reads different content on each pole (Table 28):
<table><tr><td>Pair</td><td>Real pole (L28)</td><td>Fictional pole (L28)</td></tr><tr><td>Tesla vs Vogelkirche Curie vs Brandström</td><td>Tesla, alternating, Altern, electric radio, Radio, Radiation, radioactive</td><td>Vog, v, von, Von Brand, M, H, a</td></tr></table>

Table 28: Contrastive projection at L28: the real pole reads factual associations, the fictional pole only name fragments.

The real pole reads factual associations (Tesla → alternating current, Curie → radioactivity). The fictional pole reads name fragments (Vog, von, Brand). The readout contains nothing about the entity beyond the tokens of its name.

Confirmation via entity-versus-generic contrast. Contrasting each entity against a bare frame (“A person, born in [year], invented the” / “. . . discovered”) isolates what the name adds (Table 29). The real names add factual content (Tesla: alternating current; Curie: radioactivity); the fictional names add only fragments of themselves. The hallucinated entity carries no factual content beyond its name.

Contrastive norm. The relative norm ∥∆h∥/∥h∥ at L28 is larger for real-versus-fictional pairs (mean 0.98) than real-versus-real pairs (mean 0.70). Two real entities differ less from each other than either differs from a fictional one.

Length-matched control. These entity pairs are not token-length matched: “Nikola Tesla” and “Ludvig von Vogelkirche” differ by seven tokens, so the read token sits at a different index and rotary position structure need not cancel. The first-token mismatch of §2.5 is the sharp version of this failure; a mid-prompt length offset, read at the final token, is milder, but it still warrants a check. We rebuild the contrast with real and invented names of identical token count in the same frame (Table 30). The finding holds: the real pole reads the entity’s factual associations (Tesla, alternating current; Curie, radioactivity; Newton, gravity), the fictional pole only name fragments and junk. The effect is not an artifact of the length mismatch. The contrastive-norm gap above, being a magnitude, is the claim most exposed to it, since real-versus-fictional pairs also carry the larger length differences; we read that number as suggestive rather than exact.

<table><tr><td>Entity — generic frame</td><td>Read at L28</td></tr><tr><td>Nikola Tesla (real)</td><td>alternating, Tesla</td></tr><tr><td>Ludvig von Vogelkirche (fictional)</td><td>Vog, von, Von</td></tr><tr><td>Marie Curie (real)</td><td>radio, Radio, rad</td></tr><tr><td>Helena Brandström (fictional)</td><td>Brand, brand</td></tr></table>

Table 29: Each entity minus a generic frame at L28, isolating what the name adds. Real names add factual content (Tesla, alternating current; Curie, radioactivity); fictional names add only fragments of themselves.

<table><tr><td>Real vs invented (matched length)</td><td>Real pole (L28)</td><td>Fictional pole (L28)</td></tr><tr><td>Tesla vs Kessler (11 tok)</td><td>Tesla, Altern, alternating</td><td>Kessler, tiss, helicop</td></tr><tr><td>Curie vs Kessler (10 tok)</td><td>Radio, Radiation, rad</td><td>Kessler, lake, H</td></tr><tr><td>Newton vs Kessler (10 tok)</td><td>Newton, gravity, Laws</td><td>mineral, Targ, Koen</td></tr></table>

Table 30: Length-matched control for Table 28. Each real entity is paired with an invented name (Halvard Kessler) of identical token count, same frame, read at the shared final token (Phi-2, L28). The real pole reads factual associations, the fictional pole only name fragments and junk, as in the unmatched pairs; the finding does not depend on the length mismatch.

Entropy. The real inventors predict with lower entropy (H = 1.5–2.5) than their fictional counterparts (H = 3.4–6.3), consistent with the model having specific knowledge to draw on. But entropy alone cannot separate confident recall from confident hallucination. The mountain case shows this (Tables 31 and 32). The real Mount Cook and the fictional Mount Silverhorn both predict a confident, specific height at almost the same entropy (2.14 versus 2.36), so entropy cannot tell them apart. The projection can: each real mountain minus Silverhorn reads geographic knowledge (Cook “volcano, climbers, Peaks,” Everest “Nepal, Tibet, Himalaya”), while Silverhorn reads only generic number-range priors (“1100, 1200, 1300”).

<table><tr><td>Mountain</td><td>Prompt</td></tr><tr><td>Everest (real)</td><td>Mount Everest, the tallest peak in the Himalayas, rises to</td></tr><tr><td>Cook (real)</td><td>Mount Cook, the tallest peak in New Zealand&#x27;s Southern Alps, rises to</td></tr><tr><td>Silverhorn (fictional)</td><td>Mount Silverhorn, the tallest peak in New Zealand&#x27;s Southern Alps, rises to</td></tr></table>

Table 31: Prompts for the mountain case; read at the final token (underlined).

The contrastive projection does not find a “hallucination flag.” It reads what retrieval surfaces in token space, and for fictional entities that is only name tokens and contextual priors. One caveat applies to every absence reading: a token missing from the top-K has an energy fraction f below the readout’s visibility threshold f<sup>∗</sup>, which bounds the signal but is not evidence of zero signal (Tuomi, 2026). What separates the fictional entities is that no factual content surfaces at any layer, while for the matched real entities it does.

<table><tr><td>Mountain</td><td>Entropy</td><td>Predicts</td><td>Contrastive pole</td></tr><tr><td>Mount Everest (real)</td><td>1.23</td><td>29,032 ft</td><td>Nepal, Tibet, Himalaya</td></tr><tr><td>Mount Cook (real)</td><td>2.14</td><td>3,724 m</td><td>volcano, climbers, Peaks</td></tr><tr><td>Mount Silverhorn (fictional)</td><td>2.36</td><td>≈2,800 m</td><td>1100, 1200, 1300</td></tr></table>

Table 32: The mountain case (Phi-2, “. . . rises to”). Entropy is at the next (height) token; Predicts is the greedy height. Cook (real) and Silverhorn (fictional) both emit a confident height at nearly the same entropy, so entropy cannot tell them apart: Cook recalls its real 3,724 m, Silverhorn invents one. Contrastive pole is that mountain’s side of the real-minus-fictional contrast (real poles at L24, Silverhorn at L28): the real mountains read geographic knowledge, the fictional one reads only generic number-range priors.

## 6 What the readout reads

Every reading so far has named a distinction in specific tokens: “fried, crispy” for food, “Nepal, Tibet” for a real mountain. Those tokens are the readout’s output, not the computation. So one question decides how to use them. Is a particular token identity a fact about what the model computes, or about the basis this one network happened to learn? A seed control answers it. We read four contrasts (Table 33) through five Pythia-410M models that share architecture, tokenizer, and training data and differ only in random initialization seed: the base model and four of the seed reruns released in the Pythia suite (seeds 1, 3, 6, and 7). The particular seeds are incidental to the finding, which needs only that the networks differ solely in initialization.

<table><tr><td>Contrast</td><td>Positive prompt</td><td>Negative prompt</td></tr><tr><td>Sentiment</td><td>The film was absolutely wonderful and I</td><td>The film was absolutely terrible and I</td></tr><tr><td>Temperature</td><td>I touched the metal and it felt extremely hot</td><td>I touched the metal and it felt extremely cold</td></tr><tr><td>Time</td><td>This will definitely happen tomorrow</td><td>This definitely happened yesterday</td></tr><tr><td>Nationality</td><td>She grew up in Paris speaking fluent French</td><td>She grew up in Tokyo speaking fluent Japanese</td></tr></table>

Table 33: The four contrasts read through the five seeds, each at the final token (underlined). Table 34 shows the per-seed reads.

The token-face is network-specific. At the read layer the top-10 tokens barely overlap across seeds (mean pairwise Jaccard 0.08; about 45 distinct tokens fill the 50 top-ranked slots). Yet the contrastive direction boosts the concept anchors over their opposites in every seed (20/20 seed×contrast, margins +2.8 to +5.9). Table 34 shows all four contrasts across the five seeds: the top tokens differ from seed to seed, and on this small model are often fragments, yet the concept margin stays positive throughout. Nationality reads Francophone geography (different members per seed) and time reads future words; sentiment and temperature surface mostly seed-specific fragments. The same holds across architectures. The compound-noun food reading is “delicious” in Phi-2, “burgers, cuisine” in Pythia, and “flavor, flavorful” in Qwen (§3.2). Same field, different

tokens.

<table><tr><td>Seed</td><td>Sentiment</td><td>Temperature</td><td>Time</td><td>Nationality</td></tr><tr><td>base</td><td>aug, agus</td><td> $\mathbf { t e r } , \mathsf { p s }$ </td><td>someday, unless</td><td>Alger, France</td></tr><tr><td>seed1</td><td>capturing, relaxation</td><td> $\operatorname { t e r } ,$  ting</td><td>someday, morrow</td><td>Alger, Québec</td></tr><tr><td>seed3</td><td>capture, captures</td><td> $\mathrm { t a } , \mathrm { t e }$ </td><td>tom, whichever</td><td>Alger, Belgium</td></tr><tr><td>seed6</td><td>tail, tails</td><td> $\operatorname { t e r } ,$  pone</td><td>morrow, someday</td><td>Alger, France</td></tr><tr><td>seed7</td><td>empre, adors</td><td> $\operatorname { t e r } ,$  tera</td><td>tom, dat</td><td>Alger, France</td></tr></table>

Table 34: The four contrasts read through five Pythia-410M models differing only in initialization seed (top-2 legible tokens, positive pole, at each seed’s read layer). The tokens differ from seed to seed and, on this small model, are often sub-word fragments (mean pairwise top-10 Jaccard 0.005–0.11). Yet the contrastive direction boosts the concept anchors over their opposites in every seed and contrast (20/20; margins +2.8 to +5.9): the distinction is shared even where the token-face is not. Nationality reads Francophone geography and time reads future words most legibly; sentiment and temperature surface mostly seed-specific fragments.

What this implies. A component of a write is token-shaped where it aligns with $W _ { U } ,$ and that component is real: the same distinction is recoverable in every seed. But the token basis a network lands on is a property of that trained network, not a universal code. What a computation looks like in token space is network-specific; the distinction it draws is not. The practical rule is to read the axis, not the tokens: confirm an axis by an anchor margin or an injection, not by the identity of the top-ranked tokens. This network-specific basis is one of three reasons a token can be absent from a given readout; the other two, a below-threshold magnitude and a distinction the model never emits, are limits of the readout itself, and the Discussion takes them up. A full account of how token-shaped writes superpose in the residual is beyond our scope.

## 7 Discussion

The contrastive projection reads through $W _ { U } ,$ so it sees only what the model emits into the vocabulary. Section 6 gave one reason a distinction can be missing from a readout, a network-specific basis, where the distinction is emitted but wears different tokens in each network. Two further reasons are limits of the readout itself. The first is magnitude: the aligned component sits below the visibility threshold, and averaging more baselines can raise it (Tuomi, 2026), as triangulation did for the Pythia noun read. The second is harder. A distinction the model computes but never emits into the output basis would be invisible to any $W _ { U }$ readout, however clean the contrast. Whether such computed-but-unemitted distinctions are common, and how to detect them without the readout, is an open question. It marks the ceiling of the method: contrastive projection reads emitted structure, not the full internal state.

## 7.1 A hypothesis: metaphor as a linear conceptual mapping

The metaphor readings (§4) suggest a hypothesis about the representation, beyond what four words in one model can establish. The model appears to have no single “figurativity” feature. It may instead represent a metaphorical use as the literal representation displaced by an approximately linear offset toward the metaphor’s target domain, h(metaphorical) ≈ h(literal) + v(source → target). Three observations point this way. The offset is stable across sentences for each word, a property of the mapping rather than the prompt (pairwise cosine 0.64–0.88). It is dominated by the target concept, so words that map onto the same target share it: cold and heavy, both onto emotion, correlate at cosine 0.58, while bright, onto intelligence, is orthogonal. And it can be added to induce the metaphor or subtracted to undo it, though more readily for some words (cold at unit magnitude) than others (heavy resists even at 2×). This reads as a mechanistic version of conceptual metaphor theory (Lakoff and Johnson, 1980), with the source-to-target mapping realized as a linear direction, in the spirit of linear relation decoding (Hernandez et al., 2024).

The clean test is compositional: build a target-domain direction (emotion, say) from cues unrelated to these words, and check whether injecting it induces the metaphor across new source words. If the target component is genuinely shared and separable, this should work; the graded result, where heavy resists, already suggests the separation is imperfect. We leave this to future work.

## 7.2 Future directions

Several extensions follow directly from the method.

• A developmental clock. Reading the same contrast across training checkpoints would date when a distinction becomes legible in token space, separating when a behaviour forms from when it enters the vocabulary basis. This needs a checkpointed model such as Pythia.

• An automatic visibility gate. We apply the visibility threshold f<sup>∗</sup> (Tuomi, 2026) by hand to one example (§3.1). Wiring it into the readout would flag every projection automatically, replacing the qualitative soup-versus-signal judgement with a quantitative one.

• Argument-general relation directions. The next-token contrasts here read a relation applied to a single argument. Triangulating a relation contrast over many arguments (the capital of France, of Japan, of Egypt) would cancel the argument and leave a direction that reads the relation itself. This is the natural bridge to function vectors (Todd et al., 2024): a readout of the relation direction that neither extracts nor injects a vector.

• Reading the corpus, not the capability. Because the $W _ { U }$ basis is training-specific (§3.2), the same contrast read across models probes what each model absorbed rather than what it can do. Qwen reads “hot dog” as a priced product where Phi-2 reads food. A systematic version would compare how differently-trained models frame shared concepts.

## 7.3 Limitations

• Curated pairs, not sampled. All demonstrations use hand-constructed minimal pairs. The multi-contrast triangulation uses hand-selected baselines.

• LayerNorm bypassed. We skip the final LayerNorm, so $W _ { U }$ receives vectors at the wrong scale. Token rankings are empirically invariant to this. But raw contrastive norms are not comparable across layers, because the residual-stream norm grows with depth.

$W _ { U }$ readability not guaranteed. The difference of two states was never trained for $W _ { U }$ projection. Token labels at intermediate layers are $W _ { U }$ ’s nearest-neighbour assignments. A quantitative criterion for when a component of the difference vector surfaces in this readout is developed in a companion paper (Tuomi, 2026).

• Smoothness is not $W _ { U }$ -specific. Trajectory coherence (consecutive-layer cosine) is a property of $\Delta h ,$ not of $W _ { U }$

• Exploratory, not causal. The per-position trace and per-head decomposition identify large contributors, not causes. We treat the reading as a pointer and verify causality only where we make a causal claim: activation patching for the compound-noun circuit, and the concept-axis injection (§3.1). Both interventions have known failure modes we do not rule out: a subspace can look causal without being the mechanism (Makelov et al., 2024), which bears on the conceptaxis injection, and self-repair can compensate for a patched component and mask its effect (McGrath et al., 2023), which our single-layer patches do not control for. So we report the patching as a qualitative trace, not a quantitative causal measurement. The general causal status of the difference vector is inherited from RepE, not re-established here. We ran further causal tests during this work (adding and subtracting the difference vector to steer generation across several of the contrasts above), and they behaved as expected. We omit them because they only reproduce what RepE, ActAdd, and Contrastive Activation Addition (CAA) already establish for matched-pair subtraction; nothing about steering is new here, and the paper’s contribution is the readout, not the intervention.

• Coherence is judged by inspection. Every projection returns tokens. Whether they form a coherent reading or token soup is a qualitative judgement. The visibility threshold (Tuomi, 2026) bounds when an aligned component surfaces in the top-K, but provides no automatic coherence criterion.

• Triangulation coverage. We tested multi-contrast triangulation on 6 cases. We do not know whether every model computation yields a token-readable component under contrastive subtraction, or how many baselines are sufficient in general.

• Per-position reading requires tokenization alignment. The read position must correspond to the same structural role in both inputs.

• Model coverage. Mechanistic depth is on Phi-2; the compound-noun circuit is re-run on Pythia-1.4B and Qwen2.5-1.5B (§3.2), and the seed control uses five Pythia-410M models.

## 8 Related work

Contrastive activation methods. RepE (Zou et al., 2023), ActAdd (Turner et al., 2023), and CAA (Rimsky et al., 2024) use matched-pair subtraction for steering. Du et al. (2026) introduced the operation we build on: decoding an activation difference through the logit lens, to trace meta-cognitive control in R1-style models. We develop that primitive into a systematic tracer (per-position, per-head, and sub-layer readout; multi-contrast triangulation; causal checks) across a range of contrasts and three architectures. Li et al. (2024) use contrastive pairs to study truthfulness; Ma et al. (2026) contrast contextualized and non-contextualized logits to rectify conflict-inducing layers. Both intervene; we use the contrast only to read.

Logit and tuned lens. The logit lens (nostalgebraist, 2020) and tuned lens (Belrose et al., 2023) project individual states through $W _ { U } ;$ Future Lens (Pal et al., 2023) shows a single state also carries information about tokens several positions ahead. The contrastive projection reads the content that differs between two inputs, a different subspace from the logit lens on either input alone.

Probing and linear representations. Linear probes (Belinkov, 2022) train classifiers on hidden states to detect features; Burns et al. (2023) extract truth directions from contrast pairs without supervision. The contrastive subtraction is a zero-shot linear probe along the axis the input pair defines, read out in token space rather than through a trained classifier, and triangulation extends it to arbitrary semantic axes.

Superposition and sparse autoencoders. Elhage et al. (2022) characterized superposition in toy models; Bricken et al. (2023) and Templeton et al. (2024) decompose superposed representations into monosemantic features with sparse autoencoders. We do not: triangulation cancels the content that varies across baselines and reads what is left, a reading rather than a decomposition (the surviving component may itself be a bundle, and is read only where it projects to $W _ { U } )$ . Lange et al. (2026) note that the sparsity objectives training cross-layer transcoders can reward circuits that rewrite deep computation into shallow form; our reading has no learned dictionary or sparsity penalty and so avoids that incentive, but also recovers no circuit topology and claims no faithfulness.

Circuit analysis and causal tracing. Meng et al. (2022) localized factual associations by causal tracing, Gould et al. (2024) identified successor heads from attention patterns, and Conmy et al. (2023) automated circuit discovery. Our method flags the same kind of structure (which layers, heads, and content) from the readout alone, as for the compound-noun circuit (§3.1), but does not establish causality; it is an exploratory complement to these techniques.

Factual recall and relational knowledge. Geva et al. (2023) traced factual associations to MLP layers, where the subject’s last token is enriched and the relation applied at the final position. Hernandez et al. (2024) showed many relations are well approximated by a linear map, and Todd et al. (2024) that a task or relation is carried by a compact, causal “function vector” that, added to a new context, produces the answer. The relation-versus-answer split in our next-token note (Table 7) is the observational counterpart: reading at the shared-next-token entity position surfaces the relation-conditioned content, reading at the answer position the extracted attribute. We neither extract nor inject a vector, and our per-pair difference is argument-specific, so this reads the distinction rather than recovering a function vector. The hallucination readings connect here too: for fictional entities nothing beyond name tokens surfaces.

## Use of AI Assistants

Large language models (Claude, Anthropic; Gemini, Google) were used as assistive tools for coding, running experiments, and drafting text. All research questions, experimental design, and reported claims were directed and verified by the author, who takes full responsibility for the content.

## Code and Data

Code and data are available at https://github.com/EvidentSolutions/llm-interp/tree/main/ contrastive and archived at https://doi.org/10.5281/zenodo.20843136.

## References

Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1), 2022.

Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

Trenton Bricken et al. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, Anthropic, 2023.

Collin Burns et al. Discovering latent knowledge in language models without supervision. In International Conference on Learning Representations (ICLR), 2023.

Arthur Conmy et al. Towards automated circuit discovery for mechanistic interpretability. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Yanrui Du, Yibo Gao, Sendong Zhao, Jiayun Li, Haochun Wang, Qika Lin, Kai He, Bing Qin, and Mengling Feng. From latent signals to reflection behavior: Tracing meta-cognitive activation trajectory in R1-style LLMs, 2026.

Nelson Elhage et al. Toy models of superposition. Transformer Circuits Thread, Anthropic, 2022.

Peter Gärdenfors. Conceptual Spaces: The Geometry of Thought. MIT Press, 2000.

Mor Geva et al. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Rhys Gould et al. Successor heads: Recurring, interpretable attention heads in the wild. In International Conference on Learning Representations (ICLR), 2024.

Evan Hernandez et al. Linearity of relation decoding in transformer language models. In International Conference on Learning Representations (ICLR), 2024.

George Lakoff and Mark Johnson. Metaphors We Live By. University of Chicago Press, 1980.

Georg Lange et al. Cross-layer transcoders are incentivized to learn unfaithful circuits. LessWrong, 2026.

Kenneth Li et al. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Peter Lipton. Contrastive explanation. Royal Institute of Philosophy Supplement, 27:247–266, 1990.

Xuhua Ma et al. CoRect: Context-aware logit contrast for hidden state rectification to resolve knowledge conflicts. arXiv preprint arXiv:2602.08221, 2026.

Aleksandar Makelov, Georg Lange, and Neel Nanda. Is this the subspace you are looking for? an interpretability illusion for subspace activation patching. In International Conference on Learning Representations (ICLR), 2024.

Thomas McGrath, Matthew Rahtz, János Kramár, Vladimir Mikulik, and Shane Legg. The hydra effect: Emergent self-repair in language model computations. arXiv preprint arXiv:2307.15771, 2023.

Kevin Meng et al. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

nostalgebraist. Interpreting GPT: The logit lens. LessWrong, 2020.

Charles E. Osgood, George J. Suci, and Percy H. Tannenbaum. The Measurement of Meaning. University of Illinois Press, 1957.

Koyena Pal et al. Future lens: Anticipating subsequent tokens from a single hidden state. In Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL), 2023.

Nina Rimsky et al. Steering Llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

Shangwen Sun, Alfredo Canziani, Yann LeCun, and Jiachen Zhu. The spike, the sparse and the sink: Anatomy of massive activations and attention sinks. arXiv preprint arXiv:2603.05498, 2026.

Adly Templeton et al. Scaling monosemanticity: Extracting interpretable features from Claude 3 sonnet. Transformer Circuits Thread, Anthropic, 2024.

Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function vectors in large language models. In International Conference on Learning Representations (ICLR), 2024.

Olli Tuomi. A visibility threshold for top-k logit-lens readouts. Zenodo preprint, https://doi. org/10.5281/zenodo.21461945, 2026.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023.

Bas C. van Fraassen. The Scientific Image. Oxford University Press, 1980.

Andy Zou et al. Representation engineering: A top-down approach to ai transparency, 2023.