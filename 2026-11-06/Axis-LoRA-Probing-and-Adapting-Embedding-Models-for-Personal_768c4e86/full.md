# Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval

Dohoon Kim<sup>1</sup> Junho Park<sup>1</sup> Hyunwoong Bae<sup>2</sup> Seyoon Kim<sup>2,3</sup> Taesup Moon<sup>1,2,4</sup>\*

<sup>1</sup> Department of Electrical and Computer Engineering, Seoul National University

<sup>2</sup> Interdisciplinary Program in Artificial Intelligence, Seoul National University

<sup>3</sup> Mobile eXperience Business, Samsung Electronics

<sup>4</sup> ASRI/INMC/AIIS, Seoul National University

{dohoon.kim, bryan123, hwbae0326, sykimy, tsmoon}@snu.ac.kr

## Abstract

As LLMs are increasingly asked questions about a user’s personal experiences, retrieval must distinguish memories by time, place, person, and event. Current personal-memory pipelines rely on frozen dense embedding models to embed memories for retrieval, but we find that these models do not capture the four axes well. Swapping a memory’s date or place barely shifts its embedding, and a trained projection on top of the frozen embedding does not recover the missing signal. The problem lives inside the model, where each axis is encoded by a distinct attentionhead pattern. Guided by these findings, we propose Axis-LoRA, a per-axis adapter that places four LoRA branches inside the embedding model and combines them via a per-query gate. Across three standard embedding models evaluated on LongMemEval, ATM-Bench, and LoCoMo, Axis-LoRA improves 4-benchmark mean R@10 by +1.28 to +1.56 pp while training only a small fraction of the base parameters on a small axis-counterfactual training set. Code is available at https://github. com/dohoonkim-ai/axis-lora.

## 1 Introduction

Large language models (LLMs) are increasingly expected to answer questions about a user’s personal experiences (e.g., “What did I discuss with Marco at Tuesday’s meeting?” or “Where did I park at the Boston conference?”), not just open-domain factual queries (Packer et al., 2023; Chhikara et al., 2025; Gutiérrez et al., 2024). Several benchmarks have been introduced to evaluate this personalized memory QA ability: LongMemEval (Wu et al., 2025) and LoCoMo (Maharana et al., 2024) target long-term multi-session conversations, with LoCoMo spanning multiple months, while ATM-Bench (Mei et al., 2026) extends memory to multisource personal data (images, videos, emails).

Retrieval is central to personalized memory QA: given a finite LLM context window, surfacing the query-relevant memories from a user’s long memory corpus largely determines answer quality (Lewis et al., 2020). To support retrieval over long personal memory corpora, a line of recent work develops memory-management pipelines with hierarchical and long-term storage (Chhikara et al., 2025; Kang et al., 2025; Li et al., 2025; Hu et al., 2026), yet uniformly relies on a general-purpose embedding model (Karpukhin et al., 2020; Izacard et al., 2022) for the retrieval itself.

Personal memory, however, differs in structure from general retrieval corpora: each memory is naturally indexed along a small set of categorical axes, of which we consider four: TEMPORAL (timestamps, dates), SPATIAL (places, locations), ENTITY (persons, objects), and EVENT (actions, event types). A useful personalized memory retriever must discriminate along each of these axes. General embedding model training (Karpukhin et al., 2020; Izacard et al., 2022; Gao et al., 2021; Wang et al., 2022), however, optimizes encoders for topic and content overlap, so it under-trains them on the per-memory attributes that distinguish otherwise topically similar episodic records.

We confirm this gap on three standard embedding models. Swapping a memory’s axis value (e.g., replacing Tuesday with Friday) barely changes its embedding, and the retriever therefore fails to rank the right memory near the top when a query targets that axis. To understand where the signal is missing, we run two diagnostic probes (Goldfarb-Tarrant et al., 2024). A learned projection trained on top of the frozen embedding cannot recover it, but a per-attention-head analysis shows each axis concentrated in a distinct attention-head pattern. The signal lives inside the model, not at its output.

Guided by these findings, we propose a simple per-axis adapter, Axis-LoRA (Figure 1b): four axis-conditional LoRA (Hu et al., 2022) branches inside the frozen embedding model, gated per query. We train Axis-LoRA on AxisCF, a synthetic axis-counterfactual triplet set, and evaluate its retrieval R@10 on three dense embedding models (MiniLM-L6-v2, bge-base-en-v1.5, Qwen3-Embedding-0.6B) over LongMemEval, ATM-Bench, and LoCoMo, consistently improving the 4-benchmark mean R@10 on all three embedding models.

![](images/cf9a460d05abe0a88547fda115c5a13ae0262a7810ab0fb48582cefacdaf8fd3.jpg)  
Figure 1: Overview. (a) Failure case in personal-memory retrieval. A frozen retriever groups same-content memories into a single embedding cluster, where shared content tokens place candidates at near-equal cosine distance. An axis cue in the query (here, the time axis) does not separate the gold from content-similar distractors that mismatch on the axis, so the gold falls below them in the ranked list. (b) Axis-LoRA. Four axis-conditiona LoRA adapters (T/S/E/Ev) attach to the last 20% of layers of the frozen embedding model and produce per-axis representations $\phi _ { A } ( q ) , \phi _ { A } ( m )$ . A per-query gate (MLP → sparsemax) outputs routing weights w<sub>A</sub> with exact zeros on inactive axes. The final score is the frozen-base similarity plus a gated per-axis term.

Our contributions are summarized as follows:

• We show that frozen dense encoders exhibit axis-asymmetric under-exposure of personalmemory attributes (each axis is under-exposed to a different degree), evident both in axiscounterfactual cosine distance and in axisanchored R@10.

• Two diagnostic probes localize the cause inside the encoder: an external projection over the frozen output cannot recover the gap, and each attribute draws from a distinct attentionhead footprint, which naturally motivates an inside-encoder, axis-specific fix.

• We propose Axis-LoRA, a simple per-axis adapter with four axis-conditional LoRA branches and a per-query gate, trained on AxisCF, our synthetic axis-counterfactual triplet set. On LongMemEval, ATM-Bench, and LoCoMo, Axis-LoRA improves mean R@10 across the three encoders by +1.28 to +1.56 pp, with the largest per-bench gains on the multi-evidence ATM-Bench Hard split (+2.4 to +3.8 pp across the three encoders).

## 2 Phenomenon: Axis-Asymmetric Under-Exposure

Dense embedding encoders expose the four memory axes (TEMPORAL, SPATIAL, ENTITY, EVENT) to different degrees, yet all four sit far below the cosine-distance scale required for retrieval to discriminate candidates. We characterize this axis-asymmetric under-exposure at two levels: in the embedding (§2.1) and in axis-anchored retrieval (§2.2).

Setup: Axis-isolated counterfactuals. To isolate the effect of each axis, we use AxisCF, a synthetic corpus of axis-isolated counterfactuals. For each memory M, AxisCF provides four counterfactuals $\mathrm { C F } _ { A } ,$ , one per axis $A \in$ {TEMP, SPAT, ENT, EVENT}. Each $\operatorname { C F } _ { A }$ differs from M only in tokens of axis A, leaving non-axis content unchanged (Table 1). Pool composition and axis balancing are detailed in Appendix A.3, and corpus generation in Appendix A.2.

Table 1: Axis-isolated counterfactual example. Each $\operatorname { C F } _ { A }$ swaps the value of axis A while keeping the rest of the memory unchanged.  
M I met Marco at the Boston conference on 2024-   
06-15 to discuss the rehearsal dinner.   
$\mathrm { C F } _ { \mathrm { T E M P } }$ $^ { \langle \cdot \cdot } 2 0 2 4 - 0 6 - 1 5 ^ { \prime \cdot }  ^ { \langle \cdot } 2 0 2 4 - 0 9 - 2 0 ^ { \cdot } ^ { \rangle }$   
$\mathrm { C F _ { S P A T } }$ $\mathrm { ^ { * } B o s t o n ^ { * * }  { ^ { * } S e a t t l e ^ { * * } } }$   
$\mathrm { C F } _ { \mathrm { E N T } }$ $\mathrm { \ddot { \cdot } \mathrm { \mathbf { M a r c o } ^ { 3 9 }  \mathrm { \cdots \mathrm { S a r a h } ^ { 3 9 } } } }$   
$\mathrm { C F _ { E V E N T } }$ “rehearsal dinner” → “budget meeting”

TEMPORAL SPATIAL ENTITY EVENT non-axis 1-word edit floor   
all-MiniLM-L6-v2 (22M, 384d) | floor = 0.0226   
0.5   
0.4   
0.3   
0.2   
1 一 0.1   
0.008   
0.0   
TEM SPA ENT EVE   
bge-base-en-v1.5 (110M, 768d) | floor = 0.0116   
0.3   
0.2   
0.1   
3   
0.0   
TEM SPA ENT EVE   
Qwen3-Embedding-0.6B (1024d) | floor = 0.0103   
0.4   
0.3   
0.2   
0.1 0.083   
0.0   
TEM SPA ENT EVE  
Figure 2: Axis-counterfactual cosine distance per axis across three embedding models. Violins show the distribution over the evaluation pool; diamonds mark the per-axis mean. The red dashed line is the per-model nonaxis 1-word edit floor. TEMPORAL sits at this floor on every model, while SPATIAL, ENTITY, and EVENT clear the floor but remain small in absolute magnitude.

## 2.1 Embedding-level: Axis-swap cosine response

Figure 2 plots the per-axis cosine distance 1 − cos(e(M), e(CF<sub>A</sub>)) on the pooled embeddings across three embedding models: two BERT-style encoder-only models, all-MiniLM-L6-v2 (Wang et al., 2020) (22 M) and bge-base-en-v1.5 (Xiao et al., 2024) (110 M), and one causal-attention LLM-based embedding model, Qwen3-Embedding-0.6B (Qwen Team, 2025) (600 M). As a reference scale we use a non-axis 1-word edit floor: one content word per memory swapped with a random vocabulary word, axis values excluded. We further compare against the retrieval-relevant scale $\mu _ { \mathrm { i n t e r } } ,$ the mean cosine distance between any two memories in the corpus (per-encoder values 0.5–0.8; Table 2 and Appendix B.1). All four axis swaps lie far below this scale.

Table 2: Per-model, per-axis mean axis- $\mathbf { \Omega } \cdot \mathbf { C F }$ cosine distance, expressed against two reference scales. Top: multiple of the non-axis 1-word edit floor $f _ { e } .$ Bottom: percentage of the inter-memory distance $\mu _ { \mathrm { i n t e r } } .$ . TEM-PORAL is at-floor on every model, and every axis sits $\mathrm { a t } \leq 1 4 \%$ of $\mu _ { \mathrm { i n t e r } } .$
<table><tr><td>Encoder</td><td>TEMP.</td><td>SPAT.</td><td>ENT.</td><td>EVENT</td></tr><tr><td colspan="5">Multiple of non-axis 1-word edit floor fe</td></tr><tr><td>MiniLM  $( \bar { f } _ { e } = 0 . 0 2 3 )$  bge-base</td><td> $0 . 3 4 \times$ </td><td>3.12×</td><td>3.35×</td><td>2.55×</td></tr><tr><td> $( f _ { e } = 0 . 0 1 2 )$  Qwen3-Emb-0.6B  $( f _ { e } = 0 . 0 1 0 )$ </td><td>1.12×</td><td>5.29×</td><td>5.78× 5.23×</td><td>3.01× 5.13×</td></tr><tr><td></td><td>1.04×</td><td>8.02×</td><td></td><td></td></tr><tr><td colspan="5">Percentage of inter-memory distance  $\mu _ { \mathrm { i n t e r } }$ </td></tr><tr><td>MiniLM  $( \mu { = } 0 . 7 8 )$ </td><td>1.0%</td><td>9.0%</td><td>9.7%</td><td>7.4%</td></tr><tr><td>bge-base  $( \mu { = } 0 . 4 7 )$ </td><td>2.8%</td><td>13.2%</td><td>14.4%</td><td>7.5%</td></tr><tr><td>Qwen3-Emb-0.6B  $( \mu { = } 0 . 6 5 )$ </td><td>1.6%</td><td>12.7%</td><td>68.3%</td><td>8.1%</td></tr></table>

The ordering is consistent across all three encoders: TEMPORAL is the smallest, with EVENT, SPATIAL, and ENTITY larger but all far below $\mu _ { \mathrm { i n t e r } }$ . Table 2 reports two ratios per cell. Against the non-axis 1-word edit floor, TEMPORAL is at-floor (0.3–1.1×, effectively at the noise scale across all three models), while SPATIAL, ENTITY, and EVENT clear the floor by 2.6–8.0×. Against the retrieval-relevant scale $\mu _ { \mathrm { i n t e r } }$ , however, all four axes sit at only 1–14%. The under-exposure pattern holds across all three models. It also holds beyond synthetic AxisCF memories: repeating the measurement on real memories from the three benchmarks (200 ATM-Bench, 200 LoCoMo, 100 Long-MemEval, with axis counterfactuals built by the same pipeline) reproduces the pattern, with TEM-PORAL at or below the 1-word edit floor on all three encoders and every axis $\mathrm { a t } \leq 1 1 \%$ of $\mu _ { \mathrm { i n t e r } }$ (Appendix F.5).

## 2.2 Retrieval-level: Axis-anchored R@10

We next test whether the embedding-level gap translates to retrieval failure, using an axis-anchored retrieval task: each query is a short sentence built from a single axis value of a memory M in the AxisCF evaluation pool. For example, a TEMPO-RAL query is built from M’s date as “This event happened on 2024-06-15.”, and an ENTITY query from M’s named person as “This memory involves Marco.” (per-axis templates in Appendix A.3). The retriever ranks all memories in the pool by cosine similarity to the query, with M as the gold target.

Table 3 (raw columns) reports raw cosine

Table 3: Axis-anchored retrieval R@10 on the evaluation set: raw cosine vs. projected (per-axis MLP trained with InfoNCE), with ∆ = projected − raw. TEMPORAL gets the largest projection lift, while non-TEMPORAL cells stay flat or decrease.
<table><tr><td></td><td colspan="3">MiniLM</td><td colspan="3">bge-base</td><td colspan="3">Qwen3-0.6B</td></tr><tr><td>Axis</td><td>raw</td><td>proj</td><td>∆</td><td>raw</td><td>proj</td><td>∆</td><td>raw</td><td>proj</td><td>∆</td></tr><tr><td>TEMP.</td><td>0.061</td><td>0.150</td><td>+8.9</td><td>0.072</td><td>0.167</td><td>+9.5</td><td>0.039</td><td>0.183</td><td>+14.4</td></tr><tr><td>SPAT.</td><td>0.248</td><td>0.251</td><td>+0.3</td><td>0.305</td><td>0.275</td><td>-3.0</td><td>0.231</td><td>0.233</td><td>+0.2</td></tr><tr><td>ENT.</td><td>0.163</td><td>0.224</td><td>+6.1</td><td>0.154</td><td>0.179</td><td>+2.5</td><td>0.152</td><td>0.197</td><td>+4.5</td></tr><tr><td>EVENT</td><td>0.254</td><td>0.260</td><td>+0.6</td><td>0.295</td><td>0.266</td><td>-2.9</td><td>0.312</td><td>0.231</td><td>-8.1</td></tr></table>

R@10. Across the 12 (encoder, axis) cells, R@10 ranges from 0.039 (Qwen3 TEMPORAL) to 0.312 (Qwen3 EVENT), with TEMPORAL the lowest on every encoder (0.039–0.072). Together with the embedding-level finding of §2.1, this confirms that axis information is under-exposed in frozenembedding cosine retrieval.

## 3 Diagnosis: Why Axis Signals Are Under-Exposed

The under-exposure shown in §2 raises two questions. (i) Can the missing axis discrimination be recovered from the frozen output by a learned head (i.e., does the encoder already surface it but in a geometry that cosine retrieval cannot exploit)? (ii) If not, where inside the embedding model does each axis signal live, and is the internal footprint shared across the four axes or axis-specific? Two diagnostic probes follow: a projection probe (§3.1) tests (i), and a head-spectrum probe (§3.2) tests (ii).

## 3.1 Projection probe: External projection cannot close the R@10 gap

For TEMPORAL, which sits at the non-axis edit floor, is the axis signal erased during encoding, or merely latent-compressed (preserved in the frozen output and recoverable by a learned projection)? For SPATIAL, ENTITY, and EVENT (raw cosine R@10 0.17–0.33), can an external projection improve R@10, or is this the ceiling an output-side head can reach?

Setup. We train a strong external projection: a per-axis 2-layer MLP head $\phi _ { A }$ over the frozen memory embeddings, optimized with InfoNCE. Positives are $( q _ { A } , \phi _ { A } ( m ) ) ;$ hard negatives are the axis-A counterfactuals $\operatorname { C F } _ { A }$ of m from §2. We evaluate on the same setup as §2.2. The setup isolates the question: what can we achieve with the frozen latent plus axis-CF training signal alone, without modifying the encoder? Full training and evaluation details, including two simpler projection variants we considered, are in Appendix C.1.

Two findings. (i) The axis signal is not fully erased, but recovery is small. The projection lifts TEMPORAL R@10 by +8.9 to +14.4 pp across all three encoders (Table 3), confirming that some axis information survives in the latent and is reachable by an output-side head. The absolute R@10 ceiling, however, stays at 0.15–0.18, far short of useful retrieval. An external head does not close the gap.

(ii) Projection effects vary across axes. ENTITY lifts consistently $( + 2 . 5 \mathrm { \ t o \ + 6 . 1 \mathrm { \ p p } ) }$ , while SPA-TIAL and EVENT are mixed, with per-cell deltas ranging from small positive lifts to drops of up to −8.1 pp. An output-side projection therefore does not provide a uniform fix across axes.

Implication. Axis information is partially recoverable, but a strong external head cannot bridge the R@10 gap and does not uniformly help across non-TEMPORAL axes. This rules out an outputside fix and points to modifying the embedding model itself, changing how axis information is encoded internally rather than how the frozen output is post-processed.

## 3.2 Head-spectrum probe: The attention-head spectrum is axis-asymmetric

The projection probe established that axis information lives inside the embedding model. To find where each axis signal lives, we run a per-(layer, head) analysis: for each (layer ℓ, head h) and axis A, we measure the change in the pooled per-head output when the input is perturbed by a single-edit $\mathrm { C F } _ { A }$ that alters exactly one token (Appendix C.2): $\mathrm { r e s p } [ \ell , h , A ] = \frac { \| \mathrm { h e a d } _ { \ell , h } ( M ) - \mathrm { h e a d } _ { \ell , h } ( \mathrm { C F } _ { A } ) \| _ { 2 } } { \| \mathrm { h e a d } _ { \ell , h } ( M ) \| _ { 2 } } ,$ averaged over many $( M , { \mathrm { C F } } _ { A } )$ pairs, where h $\mathrm { e a d } _ { \ell , h } ( M ) \in \mathbb { R } ^ { d _ { h } }$ is the per-head output after token-pooling (last-token for Qwen3-Embedding-0.6B, mean-pool for MiniLM and bge-base).

Axis response concentrates in late layers. Figure 3 shows the per-(layer, head) response across the three models, with brighter cells indicating heads whose pooled output changes more under axis perturbation. We find a consistent pattern: on every model and every axis, the bright cells cluster in the later layers, indicating that axis-sensitive computation is concentrated in the late layers of the embedding model.

Per-(layer, head) axis-CF response  
![](images/03eeacd2fcd61c8c3b6ae92d945f6296c77ac6eaccdec0f89262c306818a7937.jpg)  
Figure 3: Head-spectrum probe: per-(layer, head) axis response for three embedding models (rows) × four axes (columns). Within each panel, rows index transformer layers (top: input side, bottom: output side) and columns index attention heads. Brightness shows how much a head’s pooled output changes under single-edit axis perturbation, normalized per panel to a shared Low– High scale. On every model and axis, bright cells cluster in the later layers.

Table 4: Fraction of attention heads needed to cover 50% of per-axis single-edit response. A small fraction means the axis is concentrated in few heads, a large fraction that it is spread across many; the largest per model (most diffuse axis) is bolded. Within each model, TEMPORAL is the most diffuse, EVENT next, and SPATIAL and ENTITY are the most concentrated.
<table><tr><td>Model</td><td>TEMP.</td><td>EVENT</td><td>SPAT.</td><td>ENT.</td></tr><tr><td>MiniLM</td><td>41.7%</td><td>36.1%</td><td>29.2%</td><td>26.4%</td></tr><tr><td>bge-base</td><td>39.6%</td><td>35.4%</td><td>29.2%</td><td>30.6%</td></tr><tr><td>Qwen3-0.6B</td><td>32.1%</td><td>25.4%</td><td>19.6%</td><td>18.5%</td></tr></table>

Each Axis draws distinct head footprint. To quantify diffuseness, for each axis we report the percentage of top-ranked heads that together account for 50% of axis-A response, as shown in Table 4. A small percentage means the axis is concentrated in few heads, a large percentage that it is spread across many. Within each model, TEMPO-RAL is the most diffusely encoded, EVENT next, and SPATIAL and ENTITY are the most concentrated. This ordering is consistent across all three models. The same pattern is visible in Figure 3: the TEMPORAL and EVENT panels have bright cells spread across many heads, while the SPATIAL and ENTITY panels have bright cells concentrated in fewer positions. These results establish that each axis has its own response pattern across the heads of the embedding model.

## 4 Method: Axis-LoRA

The diagnosis in §3 naturally points to a simple per-axis adapter. We propose Axis-LoRA, whose three core design choices follow directly from the findings. The projection probe (§3.1) shows that an external projection over the frozen latent cannot bridge the R@10 gap, so Axis-LoRA reshapes the embedding model internally rather than projecting its output. The head-spectrum probe (§3.2) shows each axis drawing from a distinct head footprint, motivating four independent LoRA adapters. The same probe also shows axis-sensitive computation concentrated in the late layers of the embedding model, so we restrict LoRA to its late attention layers.

## 4.1 Architecture

Axis-LoRA scores a (query, memory) pair by adding a gated sum of per-axis similarities to the frozen base similarity. Each per-axis similarity is computed by a per-axis LoRA adapter inside the same embedding model, and the gate decides per query how much weight each axis carries. The retrieval score for query q and memory m is:

$$
\begin{array} { r l } & { \mathrm { s c o r e } ( q , m ) = \langle \tilde { \mathbf { z } } _ { b } ( q ) , \tilde { \mathbf { z } } _ { b } ( m ) \rangle } \\ & { \qquad + \alpha \displaystyle \sum _ { A \in \mathcal { A } } w _ { A } ( q ) \cdot \langle \tilde { \phi } _ { A } ( q ) , \tilde { \phi } _ { A } ( m ) \rangle , } \end{array}
$$

where ${ \bf z } _ { b } ( \cdot )$ is the frozen base embedding model’s pooled output, $\phi _ { A } ( \cdot )$ is the same embedding model modified by the LoRA adapter for axis $A , \ \tilde { \cdot }$ denotes ℓ<sub>2</sub>-normalization, $\boldsymbol { A } \ =$ {TEMP, SPAT, ENT, EVENT}, $w _ { A } ( q )$ is the perquery gate weight on axis A, and $\alpha \in [ 0 , 1 ]$ is the axis-term mixing weight.

Per-axis LoRA adapters. Each adapter attaches to the q\_proj and v\_proj projections of the last 20% of attention layers, with LoRA rank $r = 1 6$ The four adapters do not share parameters, so each axis can learn its own reshape direction without cross-axis interference.

Query-conditioned axis gating. A query depends on only a few axes, so the gate is designed to select them and amplify their signal: $w _ { A } ( q ) = \mathrm { s p a r s e m a x } \bigl ( \mathrm { M L P } _ { g } ( \mathbf { z } _ { b } ( q ) ) / \tau _ { g } \bigr )$ , with a single hidden-layer MLP. Sparsemax (rather than softmax) yields exact-zero entries on irrelevant axes, so unused adapters add no noise. $\tau _ { g }$ anneals from 2.0 to 0.5 over training, giving every axis early training signal before the gate sharpens.

## 4.2 Training

A single joint stage trains the four LoRA adapters and the gate MLP; the base embedding model remains frozen. The total loss combines a contrastive term on personalized memory and an anchor term on general retrieval:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { I n f o N C E } } + \lambda _ { \mathrm { K D } } \cdot \mathcal { L } _ { \mathrm { K D } } . } \end{array}
$$

Contrastive loss on AxisCF. $\mathcal { L } _ { \mathrm { { I n f o N C E } } }$ uses score $( q , m )$ from §4.1 on AxisCF, our synthetic axis-counterfactual training corpus (4,196 trips across four memory formats; construction details in Appendix A.2). Each training instance (one AxisCF triplet) consists of a positive memory $m ^ { + }$ , an anchor query $q$ taken verbatim from $m ^ { + }$ , and 1–2 single-edit axis CFs of $m ^ { + }$ (one CF per axis, with 1–2 axes sampled per instance). The batch memory pool B contains every positive and every CF in the batch, so each $\mathrm { C F } _ { A }$ of $m ^ { + }$ acts as a hard negative for $q$ and as an in-batch negative for other queries. The InfoNCE objective at temperature τ is:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \log \frac { e ^ { \mathrm { s c o r e } ( q , m ^ { + } ) / \tau } } { \sum _ { m \in \mathcal { B } } e ^ { \mathrm { s c o r e } ( q , m ) / \tau } } .
$$

Knowledge-distillation anchor on general retrieval. The axis term contributes to every query’s score at inference, including queries that lie outside the AxisCF training distribution. Without regularization the LoRAs can overfit to AxisCF and produce arbitrary embeddings on general text, injecting noise into the axis term. We anchor each LoRA to the base embedding model on a subset of MS MARCO (Bajaj et al., 2018):

$$
\mathcal { L } _ { \mathrm { K D } } = \frac { 1 } { \left| \mathcal { A } \right| } \sum _ { A \in \mathcal { A } } \mathbb { E } _ { \boldsymbol { x } \sim \mathrm { M S } \mathrm { M A R C O } } \big \| \tilde { \phi } _ { A } ( \boldsymbol { x } ) - \tilde { \mathbf { z } } _ { b } ( \boldsymbol { x } ) \big \| _ { 2 } ^ { 2 } .
$$

The anchor only acts on MS MARCO inputs, leaving the LoRAs free to specialize per axis on AxisCF under the contrastive loss.

## 5 Experiments

## 5.1 Setup

Benchmarks. We target the retrieval step in three personalized-memory QA benchmarks, covering four splits in total: LongMemEval (LME) (Wu et al., 2025), a long-term memory benchmark with multi-session chat histories (940-session memory pool); ATM-Bench (Mei et al., 2026), a multimodal personal-memory benchmark (images, videos, emails) with two splits, Normal and

Hard (Hard with more multi-evidence queries), over a ∼11K-memory pool; and LoCoMo (Maharana et al., 2024), very long-term multi-session conversations spanning multiple months (∼5.9Kturn memory pool). Per-split statistics and sample queries are in Appendix E.1. We report R@10 as the primary metric: the benchmarks pose multievidence queries (1.4–6.3 gold evidences per query on average across the four splits), so smaller cutoffs cannot reflect evidence recall. The full per-metric breakdown (R@1, R@5, R@10, R@100, MRR) is in Appendix F.1.

Models. We consider three standard dense embedding models spanning two architecture families: MiniLM-L6-v2 (Wang et al., 2020) (22 M, BERTstyle), bge-base-en-v1.5 (Xiao et al., 2024) (110 M, BERT-style), and Qwen3-Embedding-0.6B (Qwen Team, 2025) (600 M, causal-attention LLM-based). Queries are prefixed following each model’s recommended usage. For Qwen3-Embedding-0.6B, which accepts a task instruction, we use a personalmemory retrieval instruction. Full prefixes are in Appendix E.2. The main result covers all three models, while remaining experiments default to Qwen3-Embedding-0.6B.

Hyperparameters. The core Axis-LoRA configuration is shared across the three models: LoRA rank $r = 1 6$ on the last 20% of attention layers, inference mixing weight $\alpha = 0 . 2 5$ . Full per-model hyperparameters and their selection process are in Appendix E.2.

## 5.2 Main result

Table 5 reports per-benchmark R@10. Axis-LoRA consistently improves the 4-benchmark mean on every model: +1.56 pp on MiniLM-L6-v2, +1.38 pp on bge-base-en-v1.5, and +1.28 pp on Qwen3- Embedding-0.6B. The gains are means over benchmarks: two of the twelve (encoder, benchmark) cells regress (Qwen3-0.6B/ATM-Normal −1.24 pp, bge-base/LoCoMo −0.66 pp); Appendix F.3 traces both dips to a narrow rank-10 boundary effect. The largest single per-bench lift is on ATM-Bench Hard (+3.75 pp on Qwen3-Embedding-0.6B, +2.45 pp on MiniLM, +2.37 pp on bge-base), positive on all three encoders. These gains are achieved with a lightweight setup: a per-axis adapter $( \leq 0 . 6 \%$ of base parameters) trained on only 4,196 AxisCF triplets with the base embedding model frozen. All numbers are 4-seed means with cross-seed std $\leq 0 . 0 1 7$ per cell.

Table 5: R@10 across three encoders on LongMemEval (LME), ATM-Bench Normal (ATM-N), ATM-Bench Hard (ATM-H, the multi-evidence split), and LoCoMo. Each encoder block reports Base (the frozen embedding model with no adaptation, deterministic across seeds) and Axis-LoRA (mean±std over 4 random seeds), with ∆ the per-benchmark gain in pp on the seed mean; the bottom row averages across the four benchmarks. Axis-LoRA delivers +1.28 to +1.56 pp 4-bench mean R@10 across all three encoders, with the largest per-bench gain +3.75 pp on ATM-Hard (Qwen3-0.6B).
<table><tr><td></td><td colspan="3">MiniLM-L6-v2</td><td colspan="3">bge-base-en-v1.5</td><td colspan="3">Qwen3-Embedding-0.6B</td></tr><tr><td>Bench</td><td>Base</td><td>Axis-LoRA</td><td>∆</td><td>Base</td><td>Axis-LoRA</td><td>∆</td><td>Base</td><td>Axis-LoRA</td><td>∆</td></tr><tr><td>LME</td><td>0.4741</td><td>0.4872 ±0.001</td><td>+1.31</td><td>0.6099</td><td> $\mathbf { 0 . 6 3 9 7 { \scriptstyle \pm 0 . 0 0 3 } }$ </td><td>+2.98</td><td>0.7174</td><td> $\mathbf { 0 . 7 2 3 8 \pm 0 . 0 0 2 }$ </td><td>+0.64</td></tr><tr><td>ATM-N</td><td>0.6465</td><td>0.6571 ±0.002</td><td>+1.06</td><td>0.7286</td><td> $\mathbf { 0 . 7 3 6 8 \pm 0 . 0 0 0 }$ </td><td> $+ 0 . 8 2$ </td><td>0.7544</td><td> $0 . 7 4 2 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $- 1 . 2 4$ </td></tr><tr><td>ATM-H</td><td>0.2806</td><td>0.3051 ±0.003</td><td>+2.45</td><td>0.2911</td><td> $\mathbf { 0 . 3 1 4 8 \pm 0 . 0 0 8 }$ </td><td> $+ 2 . 3 7$ </td><td>0.2994</td><td> $\mathbf { 0 . 3 3 7 0 { \scriptstyle \pm 0 . 0 1 7 } }$ </td><td> $+ 3 . 7 5$ </td></tr><tr><td>LoCoMo</td><td>0.4422</td><td>0.4564±0.002</td><td>+1.42</td><td>0.6244</td><td> $0 . 6 1 7 8 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td>-0.66</td><td>0.5856</td><td> $\mathbf { 0 . 6 0 5 3 { \scriptstyle \pm 0 . 0 0 3 } }$ </td><td>+1.97</td></tr><tr><td>Avg.</td><td>0.4608</td><td> $\mathbf { 0 . 4 7 6 5 \pm 0 . 0 0 1 }$ </td><td>+1.56</td><td>0.5635</td><td> $\mathbf { 0 . 5 7 7 3 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td>+1.38</td><td>0.5892</td><td> $\mathbf { 0 . 6 0 2 0 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td>+1.28</td></tr></table>

Table 6: Query-level bootstrap 95% confidence intervals for the R@10 gains of Table 5 (20k resamples of per-query deltas, averaged over 4 seeds). All intervals exclude zero. ATM-Hard (n=31) is pooled across the three encoders. Per-split intervals are in Appendix F.2.
<table><tr><td></td><td>∆(pp)</td><td>95% CI</td></tr><tr><td>4-benchmark mean</td><td></td><td></td></tr><tr><td>MiniLM-L6-v2</td><td>+1.56</td><td> $[ + 0 . 5 4 , + 2 . 7 0 ]$ </td></tr><tr><td>bge-base-en-v1.5</td><td>+1.38</td><td> $[ + 0 . 2 5 , + 2 . 3 9 ]$ </td></tr><tr><td>Qwen3-0.6B</td><td>+1.28</td><td> $[ + 0 . 1 5 , + 2 . 5 2 ]$ </td></tr><tr><td>ATM-Hard, 3-encoder pooled</td><td>+2.86</td><td> $[ + 0 . 5 5 , + 5 . 3 6 ]$ </td></tr></table>

The 4-benchmark mean gains are statistically significant for all three encoders (Table 6). Querylevel bootstrap 95% confidence intervals (20k resamples of per-query R@10 deltas, averaged over the 4 seeds) are $[ + 0 . 5 4 , + 2 . 7 0 ]$ on MiniLM, $[ + 0 . 2 5 , + 2 . 3 9 ]$ on bge-base, and [+0.15, +2.52] on Qwen3-0.6B, all excluding zero. The ATM-Hard lift, tested pooled across the three encoders because of the small split size (n=31 queries), is likewise significant $( + 2 . 8 6 ~ \mathrm { p p ~ } [ + 0 . 5 5 , + 5 . 3 6 ] )$ . Appendix F.2 reports per-split intervals, together with means excluding ATM-Hard to check whether the gains hinge on this single small split.

## 5.3 General-retrieval preservation

We verify that Axis-LoRA preserves generalretrieval competence on MTEB (Muennighoff et al., 2023), a standard benchmark for text embedding models, evaluated on a 6-task BEIR subset (NF-Corpus, SciFact, ArguAna, SCIDOCS, FiQA2018, TRECCOVID) with Qwen3-Embedding-0.6B. The metric is nDCG@10, the standard rank-sensitive retrieval metric used in MTEB.

As shown in Table 7, mean nDCG@10 with Axis-LoRA is 0.5399 versus the base’s 0.5449, a −0.50 pp shift. The MS MARCO KD anchor keeps each LoRA branch close to the base on non-axis text without preventing the personalized-memory amplification.

Table 7: General retrieval on MTEB BEIR-6 (nDCG@10) with Qwen3-Embedding-0.6B. Axis-LoRA preserves general retrieval competence (mean $\Delta = - 0 . 5 0 \mathrm { p p } )$
<table><tr><td>Setup</td><td>NFC.</td><td>SciF.</td><td>ArgA.</td><td>SciD.</td><td>FiQA</td><td>TREC-C</td><td>Avg.</td></tr><tr><td>Base</td><td>0.3616</td><td>0.6958</td><td>0.6790</td><td>0.2305</td><td>0.4508</td><td>0.8516</td><td>0.5449</td></tr><tr><td>Axis-LoRA</td><td>0.3605</td><td>0.6924</td><td>0.6744</td><td>0.2289</td><td>0.4423</td><td>0.8410</td><td>0.5399</td></tr></table>

## 5.4 Ablation studies

Two complementary ablations isolate the contributions of Axis-LoRA’s design. The first (Table 8) compares Axis-LoRA against four design variants spanning external prompting to internal per-axis LoRA, isolating where and how axis supervision should be applied. The second (Table 9) reports leave-one-out ablations of the three remaining method components. All ablations use Qwen3-Embedding-0.6B with the same 4-bench eval pipeline; tabulated values are 4-seed means (per-cell stds in Appendix D.5).

Where to inject axis supervision. Table 8 progresses from external to internal interventions; we describe each variant alongside its measured effect.

(a) Axis-prefix prompt replaces the default Qwen instruction (§5.1) with four axis-specific instructions that direct the model to retrieve memories given a question about a particular axis (e.g., the TEMPORAL prefix begins “Given a question about WHEN (time, date, period) something happened, retrieve relevant memory entries.”; full set in Appendix E.2). The query is encoded four times, once under each axis prefix, while the memory side is encoded once with no prefix; the score for each memory is the maximum of the four per-axis cosines. We also tried mean aggregation and 5- cosine variants that include the default-prefix encoding, and all options give similar 4-bench means within ±0.002. Axis-prefix prompting yields a small lift of +0.9 pp over Base, indicating peraxis specification has a real effect. On the multievidence ATM-Hard split, however, prompting falls behind Axis-LoRA by 3.1 pp: prompt-only treatment cannot reshape the embedding enough.

(b) TF-PCA identifies axis-specific directions in the embedding space without training. For each axis A, we collect how the embedding shifts when axis A is flipped (the residual e(M) − $\mathbf { e } ( \mathrm { C F } _ { A } ) )$ across many memories, whiten these residuals, and take the top PCA components as the axis-A subspace. (c) MLP+InfoNCE replaces the training-free subspace with a per-axis 2-layer MLP head $\phi _ { A }$ trained on the same axis CFs via InfoNCE (positives $( q _ { A } , \phi _ { A } ( m ) )$ , hard negatives $( q _ { A } , \phi _ { A } ( \mathrm { C F } _ { A } ) ) )$ . At inference, both project query and memory through their four per-axis subspaces/heads and uniformly average the four peraxis cosines (gate routing requires inside-model supervision unavailable to output-side projections). The two output-side projections give only small mean R@10 changes (TF-PCA: −0.4 pp, essentially flat; MLP+InfoNCE: +0.5 pp): output-side projections cannot bridge the R@10 gap, confirming §3.1.

(d) Shared-trunk LoRA uses a single shared LoRA inside the embedding model (instead of four per-axis ones as in Axis-LoRA), followed by four axis-specific output projection heads to recover peraxis embeddings. This isolates the contribution of per-axis adaptation inside the model. Without per-axis structure inside the model, shared-trunk LoRA collapses to 0.493 (−9.6 pp below Base). Reintroducing per-axis branches with sparsemax gating (Ours) lifts mean R@10 by +10.9 pp over shared-trunk: forcing four axis-decomposed outputs through shared weights corrupts the representation, while separate per-axis branches avoid this interference.

Component contributions. Table 9 reports leave-one-out ablations of the three remaining method components: the axis-CF training signal, the KD anchor, and the sparsemax gate. Replacing axis CFs with MS MARCO negatives drops mean R@10 by −0.9 pp, confirming that the axis-CF training signal is essential. Removing the KD anchor drops mean R@10 by −1.4 pp, as the LoRAs drift on non-axis text and inject noise into the axis term at inference. Replacing the learned gate with uniform routing at inference drops mean R@10 by −0.3 pp, since uniform weighting dilutes the branches the gate has learned to emphasize. Appendix D.3 analyzes the gate’s routing behavior in detail.

Table 8: Design-variant ablation (R@10 per benchmark). Rows progress from no axis-aware treatment (Base) through axis-aware prompting, training-free and trained external projections, inside-model shared-trunk LoRA, and full Axis-LoRA. MLP+InfoNCE, Sharedtrunk LoRA, and Axis-LoRA are 4-seed means; other rows are deterministic. Per-cell stds in Appendix D.5.
<table><tr><td>Method</td><td>LME</td><td>ATM-N</td><td></td><td>ATM-H LoCoMo</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>Base</td><td>0.717</td><td>0.754</td><td>0.299</td><td>0.586</td><td>0.589</td></tr><tr><td>Axis-prefix prompt</td><td>0.723</td><td>0.748</td><td>0.308</td><td>0.603</td><td>0.596</td></tr><tr><td>TF-PCA</td><td>0.716</td><td>0.749</td><td>0.285</td><td>0.581</td><td>0.583</td></tr><tr><td>MLP+InfoNCE</td><td>0.726</td><td>0.750</td><td>0.300</td><td>0.591</td><td>0.592</td></tr><tr><td>Shared-trunk LoRA</td><td>0.635</td><td>0.611</td><td>0.255</td><td>0.469</td><td>0.493</td></tr><tr><td>Axis-LoRA</td><td>0.724</td><td>0.742</td><td>0.337</td><td>0.605</td><td>0.602</td></tr></table>

Table 9: Component ablations (R@10 per benchmark, 4-seed means) on the Axis-LoRA training signal, KD anchor, and gate. All three components are essential: removing any drops Avg. R@10 below Axis-LoRA. Percell stds in Appendix D.5.
<table><tr><td>Setting</td><td>LME</td><td>ATM-N</td><td></td><td>ATM-H LoCoMo</td><td>Avg.</td></tr><tr><td>Base</td><td>0.717</td><td>0.754</td><td>0.299</td><td>0.586</td><td>0.589</td></tr><tr><td>Axis-LoRA</td><td>0.724</td><td>0.742</td><td>0.337</td><td>0.605</td><td>0.602</td></tr><tr><td>axis-CF supervision</td><td>0.731</td><td>0.751</td><td>0.297</td><td>0.592</td><td>0.593</td></tr><tr><td> KD anchor</td><td>0.720</td><td>0.735</td><td>0.324</td><td>0.575</td><td>0.588</td></tr><tr><td>gate (uniform routing)</td><td>0.723</td><td>0.752</td><td>0.316</td><td>0.605</td><td>0.599</td></tr></table>

## 6 Related Work

Multi-aspect retrieval. Retrievers map queries and passages to a shared embedding space and are trained with contrastive objectives (Karpukhin et al., 2020; Xiong et al., 2021; Qu et al., 2021; Izacard et al., 2022; Li et al., 2023; Xiao et al., 2024). Multi-aspect retrieval extends this foundation by representing items along multiple informational aspects rather than a single semantic similarity (Kong et al., 2022; Sun et al., 2024). More recent work moves aspect specification into the query itself, through embedding models that accept a task instruction as part of the input (Lee et al., 2025a; Qwen Team, 2025). A parallel line targets a single axis directly, predominantly the temporal one (Wu et al., 2024a; Abdallah et al., 2026; Han et al.,

2025; Huynh et al., 2026). Axis-LoRA extends this line to four axes jointly, placing one adapter per axis inside the embedding model and gating them per query.

Adapter composition and routing. Low-rank adapters (Hu et al., 2022) have become a standard way to inject specialized behavior into a frozen base model, and composing or routing across multiple LoRA modules has been studied for crosstask generalization (Huang et al., 2024; Wu et al., 2024b; Kim et al., 2025). A related approach routes queries among multiple expert retrieval models (Lee et al., 2025b). Axis-LoRA composes one LoRA adapter per axis and routes across them per query.

Personalization. Personalization of retrieval and language models adapts system behavior to individual users (Zhang et al., 2025). Retrievalside approaches encode user history into embeddings (Zeng et al., 2023; Doddapaneni et al., 2024), ground retrieval in user-specific memory structures (Wang et al., 2024), or build conversational user profiles (Joko et al., 2024). Parameter-efficient fine-tuning has likewise been adapted to per-user personalization on the language-model side (Tan et al., 2024; Park et al., 2026). We instead target retrieval from a single user’s personal memory itself, where adaptation is driven by the memory content rather than by user identity.

## 7 Conclusion

We showed that frozen dense embedding models under-expose the four axes of personal-memory text in an axis-asymmetric pattern, across BERTstyle and causal-attention LLM-based architectures. Two probes localize the cause inside the model: external projection cannot bridge the R@10 gap, and each axis draws from a distinct attention-head footprint. Building on this diagnosis, we proposed Axis-LoRA, which combines four per-axis LoRA adapters inside the model, a per-query sparsemax gate, and a general-retrieval KD anchor. Axis-LoRA lifts 4-benchmark mean R@10 by +1.28 to +1.56 pp across three embedding models, with the largest per-bench gains on ATM-Hard (+2.4 to +3.8 pp across the three encoders), while training only on a small axis-counterfactual corpus.

## Limitations

Several aspects of Axis-LoRA’s scope and design are left for future work. The four axes (TEMPO-RAL, SPATIAL, ENTITY, EVENT) are tied to the benchmarks’ annotation schemas, and other domains (e.g., clinical, legal) would need their own axis schema and counterfactual construction. AxisCF’s counterfactuals edit explicit axis values; implicit cues, such as relative time expressions, vague locations, and entangled multi-entity events, are under-represented. Axis-LoRA itself is a simple per-axis baseline motivated by the diagnostic findings, leaving room for richer routing (mixture-ofexperts gates), stronger inside-model supervision, and joint training with retrieval-aware objectives. Finally, we focus on simple, data-efficient adaptation of a frozen embedding model. Whether the same recipe extends to per-user personalization, quickly fitting a retrieval model to a single user from a small amount of their own interactions, is a natural next step.

## Ethical Considerations

Our work uses personal-memory benchmarks (LongMemEval, ATM-Bench, LoCoMo) released by their original authors under appropriate privacy review; we do not collect any new user data. To check for test-set leakage, our AxisCF training corpus is screened against a contamination blacklist of named entities and date strings mined from the ATM-Bench corpus and a related personal-memory corpus (Appendix A.2). AI assistants (ChatGPT, Claude) were used for minor grammar and writing refinement.

## Acknowledgments

This work was supported in part by the National Research Foundation of Korea (NRF) grant [No. RS-2025-02263628], the Institute of Information & communications Technology Planning & Evaluation (IITP) grants [RS-2021-II212068, RS-2022- II220113, RS-2022-II220959, RS-2021-II211343], and the BK21 FOUR Education and Research Program for Future ICT Pioneers (Seoul National University), funded by the Korean government. It was also supported by Mobile eXperience (MX) Business, Samsung Electronics Co., Ltd.

## References

Abdelrahman Abdallah, Bhawna Piryani, Jonas Wallat, Avishek Anand, and Adam Jatowt. 2026. TempRetriever: Fusion-based temporal dense passage retrieval for time-sensitive questions. In Proceedings of the 19th ACM International Conference on Web Search and Data Mining (WSDM).

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A human generated MAchine reading COmprehension dataset. In arXiv preprint arXiv:1611.09268. Originally submitted 2016; final version 2018.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Sumanth Doddapaneni, Krishna Sayana, Ambarish Jash, Sukhdeep Sodhi, and Dima Kuzmin. 2024. User embedding model for personalized language prompting. In Proceedings of the First Workshop on Personalization ofGenerative AI Systems (PERSONALIZE @ EACL).

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Seraphina Goldfarb-Tarrant, Pedro Rodriguez, Jane Dwivedi-Yu, and Patrick Lewis. 2024. Multicontrievers: Analysis of dense retrieval representations. In The 7th BlackboxNLP Workshop (EMNLP).

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems (NeurIPS).

SeungYoon Han, Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, Huije Lee, and Jong C. Park. 2025. Temporal information retrieval via timespecifier model merging. In Proceedings of the 3rd Workshop on Towards Knowledgeable Foundation Models (KnowFM), pages 1–13, Vienna, Austria. Association for Computational Linguistics (ACL).

Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, and Yafeng Deng. 2026. Ever-MemOS: A self-organizing memory operating system for structured long-horizon reasoning. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (ACL).

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of

large language models. In International Conference on Learning Representations (ICLR).

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. 2024. LoraHub: Efficient cross-task generalization via dynamic LoRA composition. In Conference on Language Modeling (COLM).

Tuan-Luc Huynh, Weiqing Wang, Trung Le, Thuy-Trang Vu, Dragan Gaševic, Yuan-Fang Li, and´ Thanh-Toan Do. 2026. Efficient temporal-aware Matryoshka adaptation for temporal information retrieval. arXiv preprint arXiv:2601.05549.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research (TMLR).

Hideaki Joko, Shubham Chatterjee, Andrew Ramsay, Arjen P. de Vries, Jeff Dalton, and Faegheh Hasibi. 2024. Doing personal LAPS: LLM-augmented dialogue construction for personalized multi-session conversational search. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR).

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory OS of AI agent. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick˘ Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Dohoon Kim, Donghun Kang, and Taesup Moon. 2025. DoMIX: An efficient framework for exploiting domain knowledge in fine-tuning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), pages 14581–14602.

Weize Kong, Swaraj Khadanga, Cheng Li, Shaleen Kumar Gupta, Mingyang Zhang, Wensong Xu, and Michael Bendersky. 2022. Multi-aspect dense retrieval. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD).

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2025a. NV-Embed: Improved techniques for training LLMs as generalist embedding models. In International Conference on Learning Representations (ICLR).

Hyunji Lee, Luca Soldaini, Arman Cohan, Minjoon Seo, and Kyle Lo. 2025b. RouterRetriever: Routing over a mixture of expert embedding models. In Proceedings ofthe AAAI Conference on Artificial Intelligence.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS).

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281.

Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding Chen, Jiawei Yang, Chenyang Xi, Huayi Lai, Jihao Zhao, Yezhaohui Wang, Junpeng Ren, Zehao Lin, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhiqiang Yin, Qingchen Yu, Bo Tang, and 3 others. 2025. MemOS: An operating system for memory-augmented generation in large language models. arXiv preprint arXiv:2505.22101.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (ACL).

Jingbiao Mei, Jinghong Chen, Guangyu Yang, Xinyu Hou, Margaret Li, and Bill Byrne. 2026. According to me: Long-term personalized referential memory QA. arXiv preprint arXiv:2603.01990.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. MTEB: Massive text embedding benchmark. In Proceedings ofthe 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (EACL), pages 2014–2037.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Junho Park, Dohoon Kim, and Taesup Moon. 2026. PRISP: Privacy-safe few-shot personalization via lightweight adaptation. arXiv preprint arXiv:2601.06471.

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An optimized training approach to dense passage retrieval for opendomain question answering. In Proceedings of the 2021 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (NAACL-HLT).

Qwen Team. 2025. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Xiaojie Sun, Keping Bi, Jiafeng Guo, Sihui Yang, Qishen Zhang, Zhongyi Liu, Guannan Zhang, and Xueqi Cheng. 2024. A multi-granularity-aware aspect learning model for multi-aspect dense retrieval. In Proceedings ofthe 17th ACM International Conference on Web Search and Data Mining (WSDM).

Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. 2024. Democratizing large language models via personalized parameterefficient fine-tuning. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. MiniLM: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. In Advances in Neural Information Processing Systems (NeurIPS).

Zheng Wang, Zhongyang Li, Zeren Jiang, Dandan Tu, and Wei Shi. 2024. Crafting personalized agents through retrieval-augmented generation on editable memory graphs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations (ICLR).

Feifan Wu, Lingyuan Liu, Wentao He, Ziqi Liu, Zhiqiang Zhang, Haofen Wang, and Meng Wang. 2024a. Time-sensitive retrieval-augmented generation for question answering. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM), pages 2544– 2553.

Xun Wu, Shaohan Huang, and Furu Wei. 2024b. Mixture of LoRA experts. In International Conference on Learning Representations (ICLR).

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack: Packaged resources to advance general Chinese embedding. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR).

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations (ICLR).

Hansi Zeng, Surya Kallumadi, Zaid Alibadi, Rodrigo Nogueira, and Hamed Zamani. 2023. A personalized dense retrieval framework for unified information access. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR).

Zhehao Zhang, Ryan A. Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, Ruiyi Zhang, Jiuxiang Gu, Tyler Derr, Hongjie Chen, Junda Wu, Xiang Chen, Zichao Wang, Subrata Mitra, Nedim Lipka, and 2 others. 2025. Personalization of large language models: A survey. Transactions on Machine Learning Research (TMLR). Survey Certification.

## A AxisCF: synthetic axis-counterfactual corpus

AxisCF is the synthetic axis-controlled personalmemory corpus used both for the phenomenon analyses in $\ S 2 \mathrm { - } \ S 3$ and for training Axis-LoRA in §4. Each item is a triplet of a personal-memory text M, an anchor query q taken verbatim from M, and a set of single-edit axis counterfactuals $\{ \mathrm { C F } _ { A } \}$ where each $\mathrm { C F } _ { A }$ alters only the schema-slot values of axis A. Since M and its counterfactuals differ only in those slot values, ranking M above them concentrates the contrastive signal on exactly the under-exposed axes.

## A.1 Schema and counterfactual definition

Schema and slot edits. Each memory follows a small slot schema spanning the four axes: TEM-PORAL (date, time), SPATIAL (city, venue), EN-TITY (person, object), and EVENT (action, event type). A counterfactual $\operatorname { C F } _ { A }$ replaces the slot values of axis A with alternatives (e.g., Tuesday → Friday) while preserving every other slot verbatim. Slot values may span multiple tokens, but only one axis changes per CF; we therefore refer to these as single-edit counterfactuals in the schema-slot sense.

Per-axis counterfactual generation. For each positive memory, we generate $K = 3$ counterfactuals per axis with a prompt that constrains the change to a single axis-class. The prompt enforces token-class rules: TEMPORAL alters only dates, clock times, and year tokens $( \mathbf { e . g . , } A u g \ I 5 \to D e c$ 22); SPATIAL alters only place or venue names within a venue-type whitelist (city, hotel, restaurant, cafe, university, . . .); ENTITY alters only a single person, organization, brand, or product within a subcategory whitelist; EVENT alters only an activity term within a tight category whitelist (e.g., lunch → dinner within Meal, not lunch $ \# i g h t )$ TEMPORAL further excludes event-referring temporal expressions (e.g., Christmas, birthday) to prevent latent EVENT leakage through date words. These whitelists ensure that a single axis is varied while the rest of the schema remains semantically on-policy.

## A.2 Generation pipeline

Generator and inference settings. Positive memories and counterfactuals are both generated with Qwen3.5-27B (Qwen Team, 2026) through a vLLM endpoint, run in non-thinking mode (enable\_thinking = False) at temperature 0.7. The same model is used at temperature 0.0 for a separate validation pass (described below).

Memory format diversity. Memories are sampled across four memory formats that share the axis schema but differ in surface form:

• Short turn: a single sentence with one primary axis (TEMPORAL or EVENT).

• Multi-turn dialogue: an 8–10-turn chat covering 2–3 axes.

• Multi-field passage: an email, image log, or video transcript covering all four axes.

• Hard composition: a long-form passage, dialogue, or email thread spanning 3–4 axes.

This format mix gives surface variation while keeping the axis-controlled schema constant across forms.

Validation and filtering. Counterfactual quality is enforced at the prompt level: the generation prompt constrains each CF to a single axis-class edit under the token-class rules above, and 96% of triplets carry at least two counterfactuals per active axis. Triplets with missing counterfactuals are kept and simply train without those hard negatives, against in-batch negatives only. For the evaluationpool corpus run (Appendix A.3), a second pass with Qwen3.5-27B at temperature 0.0 additionally verifies that each counterfactual changes only the targeted axis and that the substituted value stays within the whitelist. Failing counterfactuals are discarded.

Contamination prevention. AxisCF generation is independent of the evaluation benchmarks by construction: prompts contain only the axis schema, generic category whitelists, and synthetic personas, and no benchmark text enters the generation pipeline. To audit the residual risk of incidental overlap, we screen the corpus against a contamination blacklist of named entities, location names, and date strings mined from the ATM-Bench corpus and a related personal-memory corpus. Candidate entries are collected by running spaCy en\_core\_web\_trf NER and a date regex over the test-set memories; Qwen3.5-27B then filters the candidates down to genuine named entities, and the list is normalized, deduplicated, and manually reviewed, leaving roughly 76K entries. The screen flags no benchmark-identifying entity in AxisCF: every overlap with the blacklist’s highprecision entries is a generic real-world term (major cities, hotel and electronics brands, holidays, common first names), while entities that identify specific benchmark memories (conference names, unique local venues) do not appear.

## A.3 Evaluation pool

Splits. Two disjoint splits with zero content overlap are used:

• Training split (4,196 trips): used for Axis-LoRA training (§4). A stratified 90/10 train/validation split is taken jointly over memory format and axis combination so that all 14 (format, axis-set) strata are represented in both folds.

• Fair evaluation pool (662 axis-anchored positives after a content-presence filter): used as the controlled-pool setting in §2.2 and §3.1, and re-used in Appendix D.4 for an out-ofdistribution method evaluation. This pool is generated from a separate corpus run with the same generator and axis schema as the training split, with no shared seeds or content.

Axis balancing. The fair evaluation pool is balanced across the four axes, so per-axis R@10 and per-axis cosine distances reported in §2 reflect equal-weight contributions rather than corpusfrequency artifacts.

Axis-anchored eval queries. For the axisanchored retrieval task in §2.2 and §3.1, each query is built from one schema value of a positive memory via a per-axis template:

• TEMPORAL: “This event happened on {value}.” (e.g., “This event happened on 2024- 06-15.”)

• SPATIAL: “This took place at {value}.” (e.g., “This took place at the Boston conference.”)

• ENTITY: “This memory involves {value}.” (e.g., “This memory involves Marco.”)

• EVENT: “This memory is about {value}.” (e.g., “This memory is about the rehearsal dinner.”)

A content-presence filter keeps only memories whose axis value appears verbatim in the memory text, so the retrieval target is to recover the memory by its actual axis value rather than a paraphrase. Failure on this task therefore does not reflect missing information: the value is in the gold memory’s text, but cosine retrieval compares single pooled vectors, and the value’s contribution to the memory embedding is only 1–14% of the inter-memory distance scale (§2.1). This is too small for the ranking to exploit, even though a lexical matcher would find the token directly. Lexical matching does not, however, substitute for the dense pipeline: personalmemory systems rely on embeddings for semantic content matching, and real benchmark queries often express axis cues non-verbatim (e.g., relative dates). Anchor values also recur across the pool (a median of 40–108 memories share a query’s value, depending on the axis), so even an exact-match oracle could not push R@10 close to 1. Assuming random tie-breaking among memories that contain the queried value, the oracle’s expected R@10 is roughly 0.26–0.43 depending on the axis. All four axes fall below this bound, but not equally, and TEMPORAL reaches only 9–17% of its bound.

Counterfactual filter choices. Two probes in the paper use the same AxisCF corpus but apply different counterfactual filters, suited to what each probe measures:

• §2.1 (embedding-level cosine distance) and §2.2 (axis-anchored R@10) use the full peraxis counterfactual set, which allows axisphrase swaps that may span multiple tokens (e.g., Boston conference → Seattle library for SPATIAL). The cosine and retrieval signals are sentence-level (pooled embedding), and an axis-phrase swap is a natural unit of axis change.

• §3.2 (head-spectrum probe) restricts to counterfactuals that alter exactly one schema-slot token, because per-(layer, head) responses are per-token signals that mix when multiple tokens change at once. The strict filter shrinks the SPATIAL pool to $n = 4 1$ ; a relaxed crosscheck $( n \ : = \ : 2 5 4 )$ and a position-controlled probe confirm the qualitative ordering, see Appendix C.2.

Under the strict single-token filter, the §2.1 cosine ratios shift somewhat (SPATIAL becomes a smaller multiple of the edit floor, since a singletoken swap is a smaller perturbation than a full phrase swap), but the qualitative ordering (TEM-PORAL the smallest, the other three above the floor) is unchanged.

## B Phenomenon supporting analyses

## B.1 Axis-blindness reference scales

This appendix provides the reference scales used by Figure 2 and Table 2 in §2.1.

Non-axis 1-word edit floor. For each encoder, the edit floor is the mean cosine distance between a memory M and a single-token control edit on a non-axis content word (axis values excluded). It reports the response magnitude to a near-noise content perturbation and serves as a per-encoder lower bound for “meaningful” axis response. Perencoder values: MiniLM $f _ { e } = 0 . 0 2 3$ , bge-base $f _ { e } = 0 . 0 1 2$ , Qwen3-Embedding-0.6B $f _ { e } = 0 . 0 1 0$ (Table 2, top block).

Inter-memory distance $\mu _ { \mathrm { i n t e r } } .$ For each model, $\mu _ { \mathrm { i n t e r } }$ is the mean cosine distance between any two memories in the corpus. It reflects the operating scale at which cosine retrieval distinguishes candidates: anything far below this scale will not be discriminated by cosine ranking. Per-model values: MiniLM $\mu _ { \mathrm { i n t e r } } = 0 . 7 8 .$ , bge-base $\mu _ { \mathrm { i n t e r } } = 0 . 4 7$ Qwen3-Embedding-0.6B $\mu _ { \mathrm { i n t e r } } { = } 0 . 6 5$ . Across all three models the per-axis CF cosine distances sit at 1–13% of $\mu _ { \mathrm { i n t e r } } .$ , confirming that axis swaps fall far below the inter-memory scale that determines retrieval rank.

## B.2 Robustness checks

We address three alternative explanations for the axis-asymmetric under-exposure documented in §2, and one caveat on why the phenomenon analyses use a controlled synthetic pool rather than real benchmarks.

Edit-magnitude control. A natural concern is whether TEMPORAL appears smaller simply because TEMPORAL edits modify fewer tokens than other axes. Single-edit-per-axis counterfactuals already match the schema-slot edit count, but we further normalize cosine response by Jaccard tokenedit distance. On Qwen3-Embedding-0.6B (singleedit aggregate):

Table 10: Edit-magnitude control on Qwen3- Embedding-0.6B (single-edit aggregate): per-axis cosine-response ratios vs. TEMPORAL, raw and after Jaccard token-edit normalization.
<table><tr><td>Ratio</td><td>Raw</td><td>Jaccard-normalized</td></tr><tr><td>ENTITY / TEMPORAL</td><td>4.92</td><td>1.76</td></tr><tr><td>SPATIAL / TEMPORAL</td><td>7.56</td><td>2.09</td></tr><tr><td>EVENT / TEMPORAL</td><td>4.82</td><td>1.92</td></tr></table>

TEMPORAL remains uniquely small after the strictest edit-magnitude control; roughly 25–30% of the raw asymmetry is edit-magnitude artifact, but 70–75% is encoder-intrinsic.

Position-bias control. Another concern is that TEMPORAL tokens tend to appear in late prepositional-phrase positions, and encoders may attend less strongly to such positions. We control this by generating five positional variants per memory (natural, TEMPORALas-subject, TEMPORAL-as-object, ENTITY-assubject, ENTITY-as-object) using Qwen3.5-27B, preserving the schema values, and rerun cosinedistance measurements. ENTITY-to-TEMPORAL swap-distance ratio per (model, variant):

Table 11: Position-bias control: range of ENTITYto-TEMPORAL swap-distance ratio across five positional variants (natural, TEMPORAL-as-subject, TEMPORAL-as-object, ENTITY-as-subject, ENTITYas-object).
<table><tr><td>Backbone</td><td>E / T ratio range across variants</td></tr><tr><td>MiniLM</td><td>20.4×-40.4×</td></tr><tr><td>Qwen3-0.6B</td><td>18.2×-46.2×</td></tr><tr><td>Qwen3-8B</td><td> $1 2 . 6 { \times } { - } 3 1 . 8 { \times }$ </td></tr></table>

Even when TEMPORAL is moved into the subject position, ENTITY-swap distance remains $\ge ~ 1 2 \times$ larger than TEMPORAL-swap distance. Position is at most a secondary modulator; the asymmetry is not a positional artifact.

Small-edit triviality. A third concern is that the cosine response to any small token edit is nearnoise, in which case TEMPORAL-at-floor is unremarkable. We already report the non-axis 1-word edit floor in §2.1 and Appendix B.1: TEMPORAL sits at 0.3–1.1× this floor on all three models, while SPATIAL, ENTITY, and EVENT sit at $2 . 6 { - } 8 . 0 \times$ — orders of magnitude apart, ruling out the smalledit-triviality reading.

Caveat on real-corpus axis-anchored R@10. The phenomenon analyses use a synthetic controlled pool rather than real benchmarks (Long-MemEval, ATM-Bench, LoCoMo). The reason is that LLM-extracted TEMPORAL values in real conversational corpora are systematically vaguer than other axis values. On LongMemEval, ∼ 32% of extracted TEMPORAL anchors are vague relative phrases (“next weekend”, “tonight”), ∼ 25% are month-only without year, and only ∼ 5% are full year-dates; in contrast, LongMemEval SPA-TIAL, ENTITY, and EVENT anchors are mostly specific proper nouns or distinctive activity terms. A real-corpus axis-anchored test would therefore confound encoder under-encoding (our target) with inherent natural-language vagueness of TEMPO-RAL. We use the controlled pool for the phenomenon claim, and validate end-to-end retrieval on the real benchmarks in §5.

## C Diagnostic probes: details and robustness

## C.1 Projection probe: training and full results

The projection probe in §3.1 tests whether axis information that is under-exposed in the raw cosine geometry can be recovered by an output-side head. We considered two variants of increasing capacity; the per-axis MLP with InfoNCE is the variant reported in §3.1.

Linear ridge regression. Per-axis linear map $W _ { A } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ fit by ridge regression on $( m , q _ { A } )$ pairs $( \lambda = 1 . 0 $ 5-fold trip-level CV), evaluated as $\cos ( q _ { A } , W _ { A } m )$ . This serves as a simple, closedform baseline. The raw R@10 column matches Table 3; the projected column reports ridge’s lift on the same axis-anchored evaluation. Table 12 shows that linear ridge lifts TEMPORAL only modestly (+0.6–+3.9 pp) and is destructive on SPA-TIAL and ENTITY across all three models, indicating that a single linear map cannot bridge the raw-cosine gap.

Per-axis MLP with InfoNCE (main). A per-axis 2-layer MLP $\phi _ { A }$ with shape $d  2 d $ d and

Table 12: Linear ridge variant: per-(model, axis) raw → projected R@10, ∆ = projected − raw.
<table><tr><td></td><td colspan="3">MiniLM</td><td colspan="3">bge-base</td><td colspan="3">Qwen3-0.6B</td></tr><tr><td>Axis</td><td>raw</td><td>proj</td><td>∆</td><td>raw</td><td>proj</td><td>∆</td><td>raw</td><td>proj</td><td>∆</td></tr><tr><td>TEMP.</td><td>0.061</td><td>0.082</td><td>+2.1</td><td>0.072</td><td>0.078</td><td>+0.6</td><td>0.039</td><td>0.078</td><td>+3.9</td></tr><tr><td>SPAT.</td><td>0.248</td><td>0.182</td><td>-6.6</td><td>0.305</td><td>0.226</td><td>-7.9</td><td>0.231</td><td>0.193</td><td>-3.8</td></tr><tr><td>ENT.</td><td>0.163</td><td>0.101</td><td>-6.2</td><td>0.154</td><td>0.111</td><td>-4.3</td><td>0.152</td><td>0.113</td><td>-3.9</td></tr><tr><td>EVENT</td><td>0.254</td><td>0.261</td><td>+0.7</td><td>0.295</td><td>0.328</td><td>+3.3</td><td>0.312</td><td>0.315</td><td>+0.3</td></tr></table>

GELU. The InfoNCE objective uses three negative types per axis: the positive $( q _ { A } , \phi _ { A } ( m ) )$ , the same-trip single-edit axis-A counterfactual as a hard negative $( q _ { A } , \phi _ { A } ( \mathrm { C F } _ { A } ) )$ , and in-batch othertrip memories as easy negatives. We optimize each head with Adam (learning rate $1 \times 1 0 ^ { - 4 } )$ , batch size 64, 30 epochs, and softmax temperature $\tau = 0 . 0 5$ Hyperparameters were not separately tuned per model. We use 5-fold trip-level cross-validation so each query is scored by a head trained without that trip; the evaluation protocol matches §2.2 (axis-anchored R@10 over the evaluation set). The full 12-cell results from this variant are reported in Table 3 of the body.

The MLP variant lifts TEMPORAL by $+ 8 . 9 -$ +14.4 pp (vs. +0.6–+3.9 pp for linear ridge), while non-TEMPORAL cells either stay near raw or go negative on $3 / 9$ cells. The MLP is therefore the strongest output-side probe of the two, and is the variant reported in the body.

## C.2 Head-spectrum probe: filtering and robustness

The head-spectrum probe in §3.2 uses a stricter axis-CF filter than the axis-anchored retrieval evaluation in §2.2: each $( M , { \mathrm { C F } } _ { A } )$ pair must alter exactly one schema-slot token of axis $A ,$ , not just one slot. Per-axis pair counts under this strict filter: TEMPORAL $n = 2 1 9 1$ , EVENT $n = 9 3 4$ , EN-TITY $n = 2 6 6 .$ , SPATIAL n = 41. SPATIAL is small because raw SPATIAL counterfactuals typically swap both venue and city tokens jointly, and only 41 pairs survive the strict single-token filter.

SPATIAL $n = 4 1$ cross-checks. The absolute SPATIAL coverage at $n \ = \ 4 1$ has wider sampling uncertainty than the other three axes. Two checks support the qualitative ordering (TEMPO-RAL most diffuse, the other three more concentrated):

• Rank ordering. Replacing absolute %-ofheads with its rank within the four axes per model, TEMPORAL is the most-diffuse rank-

1 in all three models, while SPATIAL, EN-TITY, and EVENT permute among ranks 2–4 without any of them dominating.

• Relaxed-CF reproducibility. Replacing the strict single-edit SPATIAL CF pool with a relaxed pool that allows venue + city joint swap $( n = 2 5 4 )$ slightly inflates the SPATIAL response but preserves the ordering: SPATIAL stays in the concentrated cluster (within 2 pp of ENTITY/EVENT %-of-heads) and TEM-PORAL stays the most diffuse.

Bootstrap 95% CIs on Qwen3-Embedding-0.6B (1000 resamples): TEMPORAL $3 2 . 1 \pm 1 . 0 \%$ EVENT 25.4 ± 1.5%, ENTITY $1 8 . 5 \pm 2 . 1 \%$ , SPA-TIAL $1 9 . 6 \pm 4 . 2 \%$ (wider, as expected from the small n). TEMPORAL is non-overlapping with the other axes at 95%.

Position-controlled probing. A natural concern is that TEMPORAL tokens may sit at canonical sequence-header positions in personal-memory data, so the diffuse-head pattern could reflect position-based attention routing. We restrict the CF pool to pairs where the perturbed token is at midsequence positions (20–100, past typical header content) and re-measure heads-to-50%-mass.

Table 13: Position-controlled head-spectrum probe: EN-TITY (E) vs. TEMPORAL (T) heads-to-50%-mass on the full CF pool vs. the mid-sequence subset (perturbed token at positions 20–100).
<table><tr><td>Model</td><td>Full pool E/T %</td><td>Mid-position pool E/T %</td></tr><tr><td>MiniLM</td><td>26.4/41.7 (ratio 0.63)</td><td>28.1/40.9 (ratio 0.69)</td></tr><tr><td>Qwen3-0.6B</td><td>18.5/32.1 (ratio 0.58)</td><td>19.4/32.4 (ratio 0.60)</td></tr></table>

The ENTITY-to-TEMPORAL ratio stays within 0.07 of the full-pool value at both models, and on Qwen3-Embedding-0.6B the TEMPORAL-vs-ENTITY gap is ≥ 13 pp at both pool definitions. The axis-asymmetric head distribution is not a positional artifact.

## D Method ablations

## D.1 α-sensitivity sweep

The α knob (axis-term mixing weight) sweeps the composite from frozen-base (α = 0) to axisonly. 4-bench mean R@10 on Qwen3-Embedding-0.6B stays at or above the frozen base across $\alpha ~ \in ~ [ 0 . 1 , 0 . 5 ]$ , with the best value 0.6020 at $\alpha \ = \ 0 . 2 5$ versus 0.5892 at $\alpha \ = \ 0 .$ The val winner (val\_axis $\mathbf { R } @ 1 0 = 0 . 9 6 5 8$ at $\alpha = 0 . 2 5 )$

Table 14: 4-seed mean±std R@10 on bge-base-env1.5. The two gate variants give comparable 4-bench means (difference 0.24 pp, within per-bench seed variance) with different per-bench trade-offs: sparsemax is slightly stronger on hard-distractor splits (LME, ATM-H), softmax+axis-CE is slightly stronger on the longercontext splits (ATM-N, LoCoMo).
<table><tr><td>Gate</td><td>LME</td><td>ATM-N</td><td>ATM-H</td><td>LoCoMo</td><td>Avg.</td></tr><tr><td>Sparsemax (Ours)</td><td>0.6397±0.003</td><td>0.7368±0.000</td><td>0.3148±0.008</td><td>0.6178±0.002</td><td>0.5773</td></tr><tr><td>Softmax + axis-CE</td><td>0.6360±0.001</td><td>0.7405 ±0.002</td><td>0.2975±0.003</td><td>0.6256 ±0.002</td><td>0.5749</td></tr></table>

coincides with the 4-bench test winner, so α is not test-tuned. Above $\alpha = 0 . 2 5 .$ , content-overlapdominant queries (ATM-Normal, LoCoMo) get over-weighted on the axis term and the mean drops, with the axis-only setting $( \alpha = 1 . 0 )$ collapsing on ATM-Normal.

## D.2 Gate variant: sparsemax vs. softmax with axis supervision

Axis-LoRA’s per-query gate uses sparsemax and is trained only by the composite contrastive objective. A natural alternative is to replace sparsemax with softmax and add an explicit axis-supervision term — for each training sample with axis label set $A ^ { * } \subseteq$ $\{ \mathrm { T } , \mathrm { S } , \mathrm { E N } , \mathrm { E V } \}$ , a cross-entropy loss between the gate weights and a uniform distribution over $A ^ { * }$ :

$$
\mathcal { L } _ { \mathrm { g a t e - C E } } = - \frac { 1 } { | A ^ { * } | } \sum _ { A \in A ^ { * } } \log w _ { A } ( q ) ,
$$

combined as $\mathcal { L } + \lambda _ { \mathrm { g a t e } } \mathcal { L } _ { \mathrm { g a t e - C E } }$ with $\lambda _ { \mathrm { g a t e } } = 0 . 0 5$ Every other hyperparameter matches the paper main configuration (§5.1). Table 14 compares the two on bge-base-en-v1.5 over the same 4 random seeds.

We keep sparsemax as the paper-main configuration because it is the simpler training recipe: it does not require per-sample axis labels and trains from axis-counterfactual pairs alone, while still producing exact zeros on inactive axes (so off-axis branches add no noise at inference).

## D.3 Gate behavior: routing collapse, a single-adapter control, and axis-supervised routing

This appendix analyzes what the learned gate of §4.1 actually does during and after training, reports a single-adapter control that isolates the source of the mean gain, and shows that a light axis supervision on the gate realizes the per-query routing that motivates the design.

The free gate collapses to one branch per encoder. Although the gate is queried per example, its learned weights degenerate: averaged over queries, a single axis branch receives nearly all routing mass, and which branch wins varies across random seeds (on Qwen3-Embedding-0.6B, the four paper seeds collapse onto EVENT, SPATIAL, ENTITY, and ENTITY respectively). The collapsed gate carries no per-query signal, giving timelabeled questions no more TEMPORAL weight than other questions (Table 16). The gate is still not vestigial at inference: on Qwen3-0.6B, routing with the collapsed learned gate beats uniform mixing over the same branches (+1.28 vs. +0.98 pp 4-benchmark mean over Base), because it selects exactly the branch that absorbed the training signal.

Why training admits correct routing but does not select it. The collapse is not due to a missing learning signal. Writing the training score as score(q, m) = $\mathrm { s i m } _ { b } ( q , m )$ + $\alpha \sum _ { A } w _ { A } ( q )$ sim $_ A ( q , m )$ , the InfoNCE gradient with respect to w<sub>A</sub>(q) is proportional to the margin sim $_ A ( q , m ^ { + } ) - \mathrm { s i m } _ { A } ( q , \mathrm { C F } )$ , i.e., to how well branch A separates the positive from the axislabeled counterfactual of that example. An axisaligned assignment, where each branch specializes on its axis and the gate routes accordingly, is therefore an optimum of the objective. It is, however, not the optimum that training reaches. All branches start identical (LoRA initialization), so the gate gradient initially distinguishes nothing; as soon as one branch pulls ahead, its margins grow on counterfactuals of every axis, the gate shifts mass to it on all examples, and the branch monopolizes the gradient. This winner-take-all dynamic mirrors the routing collapse documented for mixture-of-experts training, and the gate temperature annealing of §4.1 (designed to give every branch early signal) delays but does not prevent it: keeping the final temperature high $( \tau _ { g }  1 . 0$ instead of 0.5) still collapses onto a single axis on every seed (+1.09 ±0.34 pp).

A single-adapter control isolates the source of the mean gain. The collapsed configuration effectively scores with one jointly-trained branch. We make this exact with a control that routes every training example to a single branch (the other branches remain at the KD anchor) and scores with that branch alone, i.e., one adapter, no per-axis structure, no gate, trained on the full AxisCF set. Table 15 compares 4-benchmark mean gains over Base across the three encoders (4 seeds).

Table 15: Gate analysis, ∆ 4-benchmark mean R@10 (pp) over Base, 4-seed means. Top: single-adapter control vs. Axis-LoRA vs. the axis-supervised gate (the loss of Table 14 at $\lambda _ { \mathrm { g a t e } } { = } 0 . 0 5$ with sparsemax retained), with the supervised branches also read out under uniform routing. Bottom: routing decomposition on Qwen3-0.6B.
<table><tr><td></td><td>MiniLM bge-base</td><td></td><td>Qwen3-0.6B</td></tr><tr><td>Single adapter (control)</td><td>+1.50</td><td> $+ 1 . 1 6$ </td><td> $+ 1 . 5 9$ </td></tr><tr><td>Axis-LoRA (paper)</td><td>+1.56</td><td> ${ \bf + 1 . 3 8 }$ </td><td> $+ 1 . 2 8$ </td></tr><tr><td>+ axis-sup. gate</td><td>+1.49</td><td> $+ 0 . 9 3$ </td><td> $+ 1 . 5 4$ </td></tr><tr><td>same, uniform routing</td><td>+1.55</td><td> $+ 1 . 0 9$ </td><td> ${ \bf + 1 . 6 3 }$ </td></tr><tr><td colspan="4">Qwen3-0.6B routing decomposition</td></tr><tr><td>free gate: learned / uniform</td><td></td><td> $+ 1 . 2 8 \textit { \textsf { I } } + 0 . 9 8$ </td><td></td></tr><tr><td>axis-sup.: learned / uniform</td><td></td><td> $+ 1 . 5 4 \ / \ + 1 . 6 3$ </td><td></td></tr><tr><td>hard label routing / single adapter</td><td></td><td> $+ 1 . 3 5 \ / \ + 1 . 5 9$ </td><td></td></tr></table>

The single adapter matches the cross-encoder mean of the full method (+1.42 vs. +1.41 pp). The mean improvement is therefore driven by the axis-counterfactual training signal and adapter capacity rather than by routing per se (replacing the axis counterfactuals with generic negatives erases the full method’s gain, Table 9), consistent with the phrasing of §5.4. The gated per-axis structure matches the control at no average cost while adding what a single adapter cannot provide: encoder-level axis selection where it pays off (on bge-base the full method leads the control by +0.22 pp overall and by +0.93 pp on LongMemEval), per-axis representations (the axis-anchored evaluations of Appendix D.4 require per-axis branches), and a routing space that supervision can activate, as follows.

Axis supervision realizes per-query routing. AxisCF already labels which axis each training example edits. Adding the gate cross-entropy of Table 14 toward these labels with a small weight $( \lambda _ { \mathrm { g a t e } } = 0 . 0 5 )$ , while keeping the paper’s sparsemax gate, changes the training dynamics qualitatively: the routing mass spreads over all four branches with a seed-stable distribution (TEMPO-RAL ≈ .43, EVENT ≈ .28, ENTITY ≈ .16, SPA-TIAL ≈ .13 on Qwen3-0.6B), and genuine perquery behavior appears. Time-labeled questions receive measurably more TEMPORAL weight than other questions on LoCoMo for all three encoders and on LME for Qwen3-0.6B (Table 16); MiniLM shows no alignment on LME, the same cell where the supervised gate trails uniform routing. On Qwen3-0.6B the supervised gate lifts the mean to +1.54 (+1.63 with uniform readout) while halving the single regressing cell (ATM-Normal $- 1 . 2 4  - 0 . 6 7 ~ \mathrm { p p } )$ . Teacher-forcing the labels as hard one-hot routing, which trains each branch in isolation, recovers less of the gain (+1.35 ±0.25 $\mathrm { v s . ~ + 1 . 6 3 \pm 0 . 1 6 }$ for the soft supervised mixture), suggesting branches benefit from learning to cooperate under the mixed routing used at inference. The effect of supervision is encoder-dependent, mirroring the encoder-level specialization above: it helps Qwen3-0.6B, ties MiniLM, and trades away bge-base’s specialization advantage (+0.93 vs. +1.38 pp). We therefore report it as an analysis variant rather than a change to the method.

Table 16: Per-query alignment: mean TEMPORAL gate weight of the axis-supervised gate on time-labeled questions vs. all other questions (4-seed means). The freely trained gate shows no time-vs-other differential on any encoder (≤ .006); the supervised gate does on LoCoMo for all three encoders and on LME for Qwen3- 0.6B.
<table><tr><td></td><td colspan="2">LoCoMo (temporal, n=320)</td><td colspan="2">LME (temp.-reasoning, n=133)</td></tr><tr><td></td><td>time</td><td>other</td><td>time</td><td>other</td></tr><tr><td>MiniLM</td><td>.308</td><td>.186</td><td>.601</td><td>.603</td></tr><tr><td>bge-base</td><td>.483</td><td>.256</td><td>.653</td><td>.627</td></tr><tr><td>Qwen3-0.6B</td><td>.408</td><td>.338</td><td>.682</td><td>.637</td></tr></table>

Summary. The free gate degenerates into an encoder-level axis selector; this costs nothing on average, and its selection is what the learned-vsuniform ablation of Table 9 measures. The mean gain itself comes from training on joint four-axis counterfactuals, which a single adapter can absorb. The per-axis gated structure earns its place by matching that control on average while enabling encoder-level specialization, per-axis representations, and, under light axis supervision, the perquery routing that motivates the design.

## D.4 Axis-LoRA on the AxisCF fair eval pool

The phenomenon (§2.2) and diagnosis (§3.1) sections both evaluate on the AxisCF fair pool (662 positives, single axis-anchor + content-presence filter). For consistency we report Axis-LoRA’s R@10 on the same pool, scored with the composite $\begin{array} { r } { \langle z _ { b } ( q ) , z _ { b } ( m ) \rangle + \alpha \sum _ { A } w _ { A } ( q ) \langle \phi _ { A } ( q ) , \phi _ { A } ( m ) \rangle } \end{array}$ at $\alpha = 0 . 2 5$ . The AxisCF fair pool is generated from a separate corpus run from the training data with 0 content overlap, so this is an out-of-distribution evaluation of the trained adapter.

Table 17 reports the per-axis breakdown. Axis-LoRA improves R@10 over raw on 10/12 cells (the exceptions are MiniLM EVENT and bge-base EVENT, the latter essentially tied with raw) and improves median rank on all 12 cells. On the three cells where MLP+InfoNCE was destructive (bge-base SPATIAL/EVENT, Qwen3-Embedding-0.6B EVENT), Axis-LoRA instead lifts or matches raw (+3.0/0.0/+2.8 pp R@10), confirming that inside-encoder reshaping reaches geometry the output-side projection cannot. MiniLM lifts are smaller (the MLP+InfoNCE probe outperforms Axis-LoRA on its TEMPORAL and ENTITY cells), consistent with the small-encoder pattern in §5.2 where MiniLM’s lift on real benchmarks is also the smallest of the three encoders.

Table 17: Axis-anchored R@10 and median rank on the AxisCF fair pool: raw cosine, MLP+InfoNCE projection (§3.1), and Axis-LoRA $( \alpha = 0 . 2 5 $ , seed 42). Lower median rank = better. Axis-LoRA improves R@10 over raw on 10/12 (encoder, axis) cells and improves median rank on all 12 cells, including the three cells where MLP+InfoNCE was destructive (bge-base SPATIAL/EVENT, Qwen3-Embedding-0.6B EVENT).
<table><tr><td rowspan="2">Axis</td><td rowspan="2">Setup</td><td>MiniLM</td><td></td><td>bge-base</td><td>Qwen3-0.6B</td><td></td></tr><tr><td>R@10</td><td>med</td><td>R@10</td><td>med</td><td>R@10 med</td></tr><tr><td>TEMP</td><td>raw MLP A-LoRA</td><td>0.061 0.150 0.066</td><td>157 64 109</td><td>0.072 109 0.167 60 0.181 45</td><td>0.039 0.183 0.115</td><td>168 62 82</td></tr><tr><td>SPAT</td><td>raw MLP A-LoRA</td><td>0.248 0.250 0.304</td><td>55 57 34</td><td>0.305 29 0.275 42 0.335 26</td><td>0.231 0.233 0.282</td><td>58 48 33</td></tr><tr><td>ENT</td><td>raw MLP A-LoRA</td><td>0.163 0.224 0.190</td><td>63 65 58</td><td>0.154 0.179 0.167 82</td><td>81 0.152 73 0.197 0.196</td><td>65 49 50</td></tr><tr><td>EVENT MLP</td><td>raw A-LoRA</td><td>0.254 0.260 0.220</td><td>52 27 40</td><td>0.295 0.266 0.295 24</td><td>32 0.312 31 0.231 0.340</td><td>39 29 23</td></tr></table>

## D.5 Per-cell stds for ablation tables

Table 18 gives the 4-seed mean±std for each cell in Tables 8 and 9 that was trained (deterministic rows and the inference-time uniform-routing row are omitted).

Table 18: 4-seed mean±std for the trained rows of Tables 8 and 9.
<table><tr><td>Variant</td><td>LME</td><td>ATM-N</td><td>ATM-H</td><td>LoCoMo</td><td>Avg.</td></tr><tr><td>Axis-LoRA</td><td> $. 7 2 4 \pm . 0 0 2$ </td><td> $. 7 4 2 \pm . 0 0 3$ </td><td> $. 3 3 7 \pm . 0 1 7$ </td><td> $. 6 0 5 \pm . 0 0 3$ </td><td> $. 6 0 2 \pm . 0 0 5$ </td></tr><tr><td>Shared-trunk</td><td> $. 6 3 5 \pm . 0 2 1$ </td><td> $. 6 1 1 \pm . 0 1 1$ </td><td> $. 2 5 5 \pm . 0 0 7$ </td><td> $. 4 6 9 \pm . 0 1 4$ </td><td> $. 4 9 3 \pm . 0 0 8$ </td></tr><tr><td>MLP+InfoNCE</td><td> $. 7 2 6 \pm . 0 0 2$ </td><td> $. 7 5 0 \pm . 0 0 2$ </td><td> $. 3 0 0 \pm . 0 0 4$ </td><td> $. 5 9 1 \pm . 0 0 1$ </td><td> $5 9 2 \pm . 0 0 2$ </td></tr><tr><td>axis-CF supv.</td><td> $. 7 3 1 \pm . 0 0 4$ </td><td> $. 7 5 1 \pm . 0 0 3$ </td><td> $. 2 9 7 \pm . 0 0 3$ </td><td> $. 5 9 2 \pm . 0 0 3$ </td><td> $5 9 3 \pm . 0 0 2$ </td></tr><tr><td>KD anchor</td><td> $. 7 2 0 \pm . 0 0 4$ </td><td> $. 7 3 5 \pm . 0 0 2$ </td><td> $. 3 2 4 \pm . 0 0 7$ </td><td>.575±.007</td><td> $. 5 8 8 \pm . 0 0 4$ </td></tr></table>

Table 19: Single-axis baselines, ∆ 4-benchmark mean R@10 (pp) over Base, 4-seed means.
<table><tr><td></td><td>MiniLM</td><td>bge-base</td><td>Qwen3-0.6B</td></tr><tr><td>Temporal-only</td><td>+0.56</td><td>+0.38</td><td>-0.61</td></tr><tr><td>Temporal-only (step-matched)</td><td>+0.92</td><td>+0.77</td><td>-0.94</td></tr><tr><td>Joint 4-axis Axis-LoRA</td><td>+1.56</td><td>+1.38</td><td>+1.28</td></tr></table>

## D.6 Single-axis baselines and step matching

To verify the advantage of joint four-axis adaptation over single-axis specialization directly, we train a temporal-only baseline with the identical recipe, data pipeline, and budget as Axis-LoRA, differing only in using the TEMPORAL counterfactual triplets and no routing (temporal is the most underexposed axis in our diagnosis, so the strongest single-axis choice). Because the temporal subset is smaller (3,220 of 4,196 triplets), we additionally train a step-matched variant for the same number of optimizer steps as the joint model.

The temporal-only baseline recovers at most a third of the joint gain on MiniLM and bge-base and falls below Base on Qwen3-0.6B (Table 19); matching the optimizer steps narrows but does not close the gap, and on Qwen3-0.6B the additional temporal-only training deepens the deficit. Nor is Qwen3’s negative number an artifact of picking a weak axis: training each of the four single-axis baselines for Qwen3-0.6B under the same recipe gives −0.61 (TEMPORAL), +0.22 (SPATIAL), −0.93 (ENTITY), and −0.97 (EVENT), so even the best single axis recovers under a fifth of the joint gain. The most useful axis differs by encoder, and joint four-axis training is what makes the gain positive and consistent across encoders.

## D.7 Shared-trunk: capacity matching and training stability

Two controls confirm that the shared-trunk collapse of Table 8 reflects the design rather than optimization instability. First, a capacity-matched shared trunk with rank 64, matching the total adapter parameter count of the four rank-16 branches, still collapses under the identical recipe (0.615 ±.021 / 0.572 ±.006 / 0.248 ±.018 / 0.428 ±.008 on LME / ATM-N / ATM-H / LoCoMo, 4-bench mean 0.465 ±.009, below the rank-16 trunk’s 0.493), so added capacity does not recover performance. Second, training is stable: the smoothed contrastive loss descends steadily to a plateau and never diverges (Table 20).

The loss converges without divergence, the results are reproducible across seeds $( \mathrm { s t d s } \leq 0 . 0 2 1$ per cell), and quadrupling capacity does not help. The collapse is a stable property of forcing four axis-decomposed outputs through shared weights.

Table 20: Shared-trunk contrastive training loss (4-seed mean) at 0–100% of training steps. Both ranks converge without divergence.
<table><tr><td></td><td>0%</td><td>20%</td><td>40%</td><td>60%</td><td>80%</td><td>100%</td></tr><tr><td>Shared trunk, r=16</td><td>0.90</td><td>0.75</td><td>0.65</td><td>0.60</td><td>0.51</td><td>0.45</td></tr><tr><td>Shared trunk, r=64</td><td>0.77</td><td>0.64</td><td>0.50</td><td>0.58</td><td>0.52</td><td>0.55</td></tr></table>

## E Experimental setup

## E.1 Evaluation benchmark details

Table 21 summarizes per-split statistics for the four evaluation splits referenced in §5.1.

Table 21: Per-split statistics. Pool is the memory pool the retriever ranks over. Avg evid. is the mean number of gold supporting memories per query; Multi-evid. is the fraction of queries with more than one gold memory.
<table><tr><td>Benchmark</td><td>Split</td><td>Queries</td><td>Pool</td><td>Avg evid.</td><td>Multi-evid.</td></tr><tr><td>LongMemEval</td><td></td><td>500</td><td>940</td><td>1.90</td><td>65%</td></tr><tr><td>LoCoMo</td><td></td><td>1,977</td><td>5,882</td><td>1.42</td><td>21%</td></tr><tr><td>ATM-Bench</td><td>Normal</td><td>1,013</td><td>11,034</td><td>1.44</td><td>25%</td></tr><tr><td>ATM-Bench</td><td>Hard</td><td>31</td><td>11,034</td><td>6.26</td><td>100%</td></tr></table>

Sample queries. One representative query per split:

• LongMemEval: “[2023/04/10] What was the first issue I had with my new car after its first service?”

• LoCoMo: “When did Caroline go to the LGBTQ support group?”

• ATM-Bench Normal: “Which hotel did I stay in on the first day of my trip to Kent?”

• ATM-Bench Hard: “I really like the steel bridge in Porto. Can you help me find all the images and videos I took during my visit?”

## E.2 Per-model training hyperparameters

Shared method hyperparameters. The following hyperparameters are identical across all three embedding models: LoRA rank r = 16 applied to the last 20% of attention layers on q\_proj and v\_proj with dropout 0.05; InfoNCE temperature $\tau ~ = ~ 0 . 0 5 ;$ gate temperature annealed $\tau _ { g } : 2 . 0  0 . 5 ; \mathrm { { K D } }$ weight $\lambda _ { \mathrm { K D } } ~ = ~ 0 . 3$ over a 50K MS MARCO triplet pool; inference mixing weight α = 0.25; AdamW with adapter learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay 10<sup>−4</sup>, and warmup ratio 0.05.

Selection process. The inference mixing weight α is selected via a documented sweep on Qwen3- Embedding-0.6B (Appendix D.1). LoRA rank (r = 16) and depth (last 20% of attention layers) were chosen by a small grid on a single seed of Qwen3-Embedding-0.6B; rank expansion from r = 8 to r = 16 added +0.30 pp 4-bench mean at 76% more trainable parameters (still 0.26% of base), and depth expansion was not explored due to GPU memory limits. The InfoNCE temperature and AdamW with linear warmup are standard defaults from dense-retrieval training. The gate annealing schedule, KD weight, and adapter learning rate were set by small grid sweeps on Qwen3- Embedding-0.6B during development (a 3×3 grid over the KD weight and the final gate temperature, plus learning-rate sweeps), selected by R@10 on the held-out AxisCF validation split. Per-model optimization hyperparameters (max sequence length, batch size, epochs) are chosen to fit each architecture’s memory footprint and training dynamics. Everything else is selected on Qwen3-Embedding-0.6B and transferred to MiniLM-L6-v2 and bgebase-en-v1.5 without retuning.

Query instruction prefixes. Each model is evaluated with its model-card-recommended query prefix: MiniLM-L6-v2 uses no prefix; bge-base-en-v1.5 uses “Represent this sentence for searching relevant passages: ”; Qwen3- Embedding-0.6B uses “Instruct: Given a personalmemory question, retrieve relevant memory entries.\nQuery: ”, a personal-memory-targeted instantiation of Qwen3’s task-instruction interface.

Axis-specific instructions (axis-prefix prompt baseline). The Axis-prefix prompt variant in Table 8 replaces the default Qwen instruction with four axis-specific instructions: TEMPORAL “Instruct: Given a question about WHEN (time, date, period) something happened, retrieve relevant memory entries.”; SPATIAL “Instruct: Given a question about WHERE (location, place) something happened, retrieve relevant memory entries.”; ENTITY “Instruct: Given a question about WHO or WHAT (people, objects) is involved, retrieve relevant memory entries.”; EVENT “Instruct: Given a question about WHAT EVENT or ACTION occurred, retrieve relevant memory entries.”. All four are followed by “\nQuery: ”.

Software environment. All reported evaluations were produced with a single pipeline under Py-Torch 2.10.0 (CUDA 12.8), Transformers 5.6.2, and PEFT 0.19.1.

## E.3 MTEB BEIR-6 reproduction setup

Table 7 reports per-task nDCG@10 for Qwen3- Embedding-0.6B base (α = 0) and + Ours (α = 0.25) on the BEIR-6 subset (NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA2018, TRECCOVID). Both rows use the same evaluation configuration: max-len-q 256, max-len-m 512, and the generalretrieval query instruction prefix (“Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: ”). Our absolute base scores (mean 0.5449) are below the official MTEB leaderboard report for Qwen3-Embedding-0.6B (mean 0.5882): the gap is dominated by max-length truncation on the long-text tasks, since the official run uses task-specific instructions and a longer maxlength budget. The relative ∆ between the two rows is unaffected because both rows share the same shortened pipeline; this is the comparison we report in §5.2.

Cross-encoder MTEB (MiniLM, bge-base). Table 23 reports the same BEIR-6 sweep on the other two main-table encoders, each evaluated with its model-card-recommended query prefix (MiniLM: none; bge-base: “Represent this sentence for searching relevant passages:”).

The three embedding models show encodersize-graded preservation: MiniLM (22M) shows the largest mean degradation (−1.10 pp), concentrated on TRECCOVID (−4.78 pp) and FiQA2018 (−1.53 pp); bge-base (110M) is near-perfect (−0.02 pp), with an ArguAna gain (+1.90 pp) partly offsetting modest drops on long-text splits; Qwen3-Embedding-0.6B (600M) sits in between at −0.50 pp (Table 7), with the largest single drop on TRECCOVID (−1.06 pp). The smaller MiniLM has proportionally larger LoRA influence (last 20% covers 6/12 layers versus 5/28 on Qwen), so the same axis-conditional reshaping translates to a larger perturbation of the general-retrieval geometry. The personalized-memory gains (+1.28 to +1.56 pp 4-bench mean R@10) more than offset the BEIR mean drops on every embedding model, but the trade-off is real and should be considered when transferring the recipe to small encoders.

Table 22: Per-architecture optimization hyperparameters. Shared method hyperparameters are listed in the surrounding text.
<table><tr><td></td><td>MiniLM-L6-v2</td><td>bge-base-en-v1.5</td><td>Qwen3-Embedding-0.6B</td></tr><tr><td>Max sequence length</td><td>512</td><td>512</td><td>1,536</td></tr><tr><td>Per-GPU batch size</td><td>64</td><td>16</td><td>2</td></tr><tr><td>Gradient accumulation</td><td>1</td><td>1</td><td>2</td></tr><tr><td>Effective batch (×4 GPU)</td><td>256</td><td>64</td><td>16</td></tr><tr><td>Epochs</td><td>100</td><td>5</td><td>5</td></tr><tr><td>Learning rate (gate)</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>MS MARCO per batch</td><td>8</td><td>8</td><td>2</td></tr></table>

Table 23: General retrieval (MTEB BEIR-6, nDCG@10) on MiniLM-L6-v2 and bge-base-en-v1.5. The two BERT-style embedding models show different preservation: bge-base near-perfect (−0.02 pp) and MiniLM mildly degraded (−1.10 pp).
<table><tr><td>Setup</td><td>NFC.</td><td>SciF.</td><td> $\mathbf { A r g A } .$ </td><td>SciD.</td><td>FiQA</td><td>TREC-C |</td><td> $\mathbf { A v g } .$ </td><td>∆</td></tr><tr><td colspan="9">MiniLM-L6-v2</td></tr><tr><td>Base</td><td>0.3152</td><td>0.6452</td><td>0.4949</td><td>0.2172</td><td>0.3681</td><td>0.4748</td><td>0.4192</td><td></td></tr><tr><td>Axis-LoRA</td><td>0.3133</td><td>0.6555</td><td>0.4849</td><td>0.2161</td><td>0.3528</td><td>0.4270</td><td>0.4083</td><td>-1.10</td></tr><tr><td colspan="9">bge-base-en-v1.5</td></tr><tr><td>Base</td><td>0.3773</td><td>0.7443</td><td>0.6026</td><td>0.2219</td><td>0.4002</td><td>0.7672</td><td>0.5189</td><td></td></tr><tr><td>Axis-LoRA</td><td>0.3747</td><td>0.7501</td><td>0.6216</td><td>0.2174</td><td>0.3951</td><td>0.7538</td><td>0.5188</td><td>-0.02</td></tr></table>

## F Additional results

## F.1 Per-metric breakdown

Table 24 gives the full 4-seed per-metric breakdown (R@1, R@5, R@10, R@100, MRR) for the main 4-benchmark result of Table 5, on each (model, benchmark) cell.

Rank-redistribution mechanism. On Qwen3- Embedding-0.6B, R@5 dips slightly on 4-bench mean while R@10 lifts +1.28 pp; R@100 and MRR are essentially flat. Together these describe a rank-redistribution mechanism: the additive axis term reranks within the cosine-near band where §3.1 identified residual axis signal that base cosine alone cannot rank. Rank-11 to 20 axis-matching candidates move into the top-10. The pattern is sharpest on ATM-Hard (+3.75 pp R@10), consistent with multi-evidence queries benefiting most from per-axis ranking precision in the cosine-near band. On MiniLM-L6-v2 the gains are uniformly positive across all five aggregate metrics: the smallest model’s baseline axis exposure is weak across all four axes (Table 2), so Axis-LoRA lifts every retrieval band without rank redistribution. On bge-base-en-v1.5 the lift is concentrated on 3/4 benchmarks (LME +2.98, ATM-Normal +0.82, ATM-Hard +2.37) with a small dip on LoCoMo (−0.66 pp), giving a 4-bench mean of +1.38 pp.

## F.2 Bootstrap confidence intervals

All intervals are query-level bootstraps: the perquery R@10 delta (Axis-LoRA − Base), averaged over the 4 seeds, is resampled 20k times; 4-benchmark means use a hierarchical resample within each benchmark. Table 25 reports, alongside the 4-benchmark mean intervals quoted in §5.2, the ATM-Hard split alone, the mean over the other three benchmarks, and the encoder-pooled ATM-Hard gain (per query, the gain is first averaged over the three encoders, then the 31 queries are resampled jointly, preserving the correlation from evaluating all encoders on the same queries).

## F.3 Per-query analysis of the regressing cells

Two of the twelve (encoder, benchmark) cells regress (Qwen3-0.6B/ATM-Normal −1.24 pp, bgebase/LoCoMo −0.66 pp). Per-query ranks over the 4 seeds (Table 26) show the dips are a narrow rankboundary effect rather than broad degradation: on both cells the large majority of queries keep the same top-10 outcome, and the net change comes from a small set of golds crossing the rank-10 cutoff, the rank-redistribution mechanism discussed with Table 24. In these two cells the exchange happens to be slightly unfavorable, while in the other ten it is favorable.

## F.4 Query-type-stratified gains on real benchmarks

Using each benchmark’s own question-type labels, only one type per dataset isolates a single axis, the TEMPORAL axis (temporal-reasoning in LongMemEval, temporal in LoCoMo); the other types blend several axes, so a gain there cannot be attributed to one axis. Across the six encoder×benchmark cells for time questions, every encoder improves significantly on time questions in at least one of the two benchmarks, and no cell shows a significant loss (Table 27), matching the axis the diagnosis flags as most under-exposed.

Table 24: Per-metric breakdown of the main 4-benchmark result (4-seed mean±std, α = 0.25). ∆ = Ours − Base in pp.
<table><tr><td></td><td></td><td colspan="3">R@1</td><td colspan="3">R@5</td><td colspan="3">R@10</td><td colspan="3">R@100</td><td colspan="3">MRR</td></tr><tr><td>Encoder</td><td>Bench</td><td>Base</td><td>Ours</td><td>∆</td><td>Base</td><td>Ours</td><td>∆</td><td>Base</td><td>Ours</td><td>∆</td><td>Base</td><td>Ours</td><td>∆</td><td>Base</td><td>Ours</td><td>∆</td></tr><tr><td rowspan="4">MiniLM</td><td>LME</td><td>.198</td><td>.203±.002</td><td>+0.4</td><td>.378</td><td>.398±.002</td><td>+2.1</td><td>.474</td><td> $. 4 8 7 \pm . 0 0 1$ </td><td>+1.3</td><td>.816</td><td> $. 8 1 9 { \pm } . 0 0 2 $ </td><td>+0.3</td><td>.382</td><td> $. 3 9 2 { \scriptstyle \pm . 0 0 4 }$ </td><td>+1.1</td></tr><tr><td>ATM-N</td><td>.352</td><td>.352±.002</td><td>+0.1</td><td>.566</td><td>.576±.003</td><td>+1.0</td><td>.647</td><td> $. 6 5 7 { \scriptstyle \pm . 0 0 2 }$ </td><td>+1.1</td><td>.849</td><td> $. 8 5 4 \pm . 0 0 1$ </td><td>+0.5</td><td>.501</td><td> $. 5 0 7 { \scriptstyle \pm . 0 0 1 }$ </td><td>+0.5</td></tr><tr><td>ATM-H</td><td>.037</td><td>.059±.002</td><td>+2.2</td><td>.224</td><td>.238±.003</td><td>+1.4</td><td>.281</td><td> $. 3 0 5 { \pm } . 0 0 3$ </td><td>+2.5</td><td>.633</td><td>.659±.004</td><td>+2.5</td><td>.328</td><td> $. 3 9 6 { \pm } . 0 1 0$ </td><td>+6.8</td></tr><tr><td>LoCoMo</td><td>.154</td><td>.157±.001</td><td>+0.4</td><td>.350</td><td>.360±.002</td><td>+1.0</td><td>.442</td><td> $. 4 5 6 { \pm } . 0 0 2 $ </td><td>+1.4</td><td>.772</td><td>.777±.002</td><td>+0.5</td><td>.284</td><td> $. 2 8 9 \pm . 0 0 1$ </td><td>+0.5</td></tr><tr><td rowspan="4">bge-base</td><td>LME</td><td>.233</td><td>.254±.003</td><td>+2.1</td><td>.481</td><td>.519±.003</td><td>+3.8</td><td>.610</td><td> $. 6 4 0 { \pm } . 0 0 3 $ </td><td>+3.0</td><td>.877</td><td>.895±.000</td><td>+1.8</td><td>.474</td><td> $. 5 0 5 { \scriptstyle \pm . 0 0 2 }$ </td><td>+3.1</td></tr><tr><td>ATM-N</td><td>.435</td><td>.437±.002</td><td>+0.2</td><td>.663</td><td>.664±.002</td><td>+0.1</td><td>.729</td><td>.737±.000</td><td>+0.8</td><td>.898</td><td>.898±.002</td><td>+0.0</td><td>.597</td><td> $. 6 0 1 { \scriptstyle \pm . 0 0 1 }$ </td><td>+0.4</td></tr><tr><td>ATM-H</td><td>.069</td><td>.065±.003</td><td>-0.3</td><td>.237</td><td>.221±.003</td><td>-1.6</td><td>.291</td><td>.315±.008</td><td>+2.4</td><td>.650</td><td>.669±.007</td><td>+1.9</td><td>.394</td><td> $. 4 0 7 { \scriptstyle \pm . 0 1 2 }$ </td><td>+1.3</td></tr><tr><td>LoCoMo</td><td>.277</td><td>.268±.002</td><td>-0.9</td><td>.518</td><td>.517±.002</td><td>-0.1</td><td>.624</td><td>.618±.002</td><td>-0.7</td><td>.878</td><td>.880±.001</td><td>+0.2</td><td>.436</td><td> $. 4 2 8 { \pm } . 0 0 1$ </td><td>-0.8</td></tr><tr><td rowspan="4">Qwen3-0.6B</td><td>LME</td><td>.291</td><td>.298±.004</td><td>+0.7</td><td>.617</td><td>.616±.005</td><td>-0.1</td><td>.717</td><td>.724±.002</td><td>+0.6</td><td>.940</td><td>.940±.001</td><td>-0.0</td><td>.579</td><td> $. 5 8 8 { \scriptstyle \pm . 0 0 4 }$ </td><td>+0.9</td></tr><tr><td>ATM-N</td><td>.435</td><td>.435±.002</td><td>+0.0</td><td>.679</td><td>.666±.005</td><td>-1.3</td><td>.754</td><td>.742±.003</td><td>-1.2</td><td>.906</td><td>.905±.003</td><td>-0.1</td><td>.600</td><td> $. 5 9 3 { \pm } . 0 0 2$ </td><td>-0.7</td></tr><tr><td>ATM-H</td><td>.075</td><td>.056±.008</td><td>-1.9</td><td>.228</td><td>.188±.007</td><td>-3.9</td><td>.299</td><td>.337±.017</td><td>+3.8</td><td>.747</td><td>.762±.006</td><td>+1.5</td><td>.416</td><td> $. 3 8 3 \pm . 0 1 7$ </td><td>-3.3</td></tr><tr><td>LoCoMo</td><td>.273</td><td>.280±.003</td><td>+0.6</td><td>.508</td><td>.519±.002</td><td>+1.1</td><td>.586</td><td>.605±.003</td><td>+2.0</td><td>.836</td><td>.854±.001</td><td>+1.9</td><td>.426</td><td> $. 4 3 6 { \pm } . 0 0 2 $ </td><td>+1.0</td></tr></table>

Table 25: Query-level bootstrap 95% CIs for R@10 gains (pp). At n=31 no single ATM-Hard interval excludes zero, reflecting the sample size; the gain is positive for all three encoders and significant when pooled across them.
<table><tr><td></td><td colspan="2">ATM-Hard (n=31)</td><td colspan="2">3-bench mean (excl. ATM-H)</td><td colspan="2">4-bench mean</td></tr><tr><td></td><td>∆</td><td>95% CI</td><td>Δ</td><td>95% CI</td><td>Δ</td><td>95% CI</td></tr><tr><td>MiniLM</td><td>+2.45</td><td>[−0.85, +6.44]</td><td>+1.26</td><td> $[ + 0 . 5 5 , + 1 . 9 9 ]$ </td><td>+1.56</td><td> $[ + 0 . 5 4 , + 2 . 7 0 ]$ </td></tr><tr><td>bge-base</td><td>+2.37</td><td>[−1.42, +5.74]</td><td>+1.05</td><td> $[ + 0 . 3 0 , + 1 . 8 2 ]$ </td><td>+1.38</td><td> $[ + 0 . 2 5 , + 2 . 3 9 ]$ </td></tr><tr><td>Qwen3-0.6B</td><td>+3.75</td><td>[−0.35, +8.48]</td><td>+0.45</td><td> $[ - 0 . 1 6 , + 1 . 0 6 ]$ </td><td>+1.28</td><td> $[ + 0 . 1 5 , + 2 . 5 2 ]$ </td></tr><tr><td>3-encoder pooled +2.86</td><td></td><td>[+0.55, +5.36]</td><td></td><td></td><td></td><td></td></tr></table>

Table 26: Per-query anatomy of the two regressing cells (4-seed averages). “Unchanged” = same top-10 hit outcome under Base and Axis-LoRA; “wins/losses” = golds that newly enter / drop out of the top-10 per seed; ranks are medians before → after.
<table><tr><td>Cell</td><td>Unchanged</td><td></td><td>Wins / losses Winner rank Loser rank</td><td></td></tr><tr><td>Qwen3 × ATM-N 92% (933/1,013)</td><td></td><td>27/50</td><td> $1 2 \to 7$ </td><td> $6  1 2$ </td></tr><tr><td>bge × LoCoMo</td><td> $9 1 \% \ ( \mathrm { { i } } , 7 9 \dot { 4 } / 1 , 9 7 7 )$ </td><td>80 /103</td><td> $1 5  6$ </td><td> $7  1 2$ </td></tr></table>

## F.5 Axis-response replication on real benchmark memories

ATM-Bench labels by answer format rather than axis and cannot be split this way.

We repeat the axis-response measurement of Table 2 on real memories drawn from the three evaluation benchmarks (200 ATM-Bench, 200 LoCoMo, 100 LongMemEval), building axis counterfactuals with the same generation pipeline and using the same 1-word content-edit floor. The pattern reproduces with the same axis ordering (Table 28): TEMPORAL sits at or below the edit floor on all three encoders, the other axes clear the floor, and every axis stays at or below 11% of the inter-memory distance, far below the scale at which cosine ranking discriminates. The pattern also holds within each source benchmark.

Table 27: Time-question R@10 gains (pp) with querylevel bootstrap 95% CIs (4 seeds). Bold intervals exclude zero.
<table><tr><td></td><td>temporal-reasoning  $( \mathrm { L M E } , n { = } 1 3 3 )$ </td><td>temporal  $( \mathrm { L o C o M o } , n { = } 3 2 0 )$ </td></tr><tr><td>MiniLM</td><td>+1.32  $[ - 2 . 4 , + 5 . 0 ]$ </td><td>+2.99  $[ + \mathbf { 0 . 8 } , + \mathbf { 5 . 3 } ]$ </td></tr><tr><td>bge-base</td><td> ${ \bf + 4 . 4 6 }$   $[ + \mathbf { 1 . 1 } , + \mathbf { 8 . 0 } ]$ </td><td> $- 0 . 6 5$  -  $\cdot 3 . 7 , + 2 . 2 ]$ </td></tr><tr><td>Qwen3-0.6B</td><td> $+ 0 . 1 1$   $[ - 2 . 5 , + 2 . 7 ]$ </td><td>+3.02  $[ + \mathbf { 1 . 0 } , + \mathbf { 5 . } \dot { \mathbf { 2 } } ]$ </td></tr></table>

Table 28: Axis-swap response on real benchmark memories: multiple of the 1-word edit floor / percentage of the inter-memory distance.
<table><tr><td></td><td>TEMPORAL</td><td>SPATIAL</td><td>ENTITY</td><td>EVENT</td></tr><tr><td>MiniLM</td><td>0.05× / 0.2%</td><td> $1 . 4 1 \times / 5 . 2 \%$ </td><td> $2 . 5 7 \times / 9 . 6 \%$ </td><td> $1 . 6 4 \times / 6 . 1 \%$ </td></tr><tr><td>bge-base</td><td> $0 . 2 8 \times / 1 . 1 \%$ </td><td> $1 . 5 7 \times / 6 . 3 \%$ </td><td>1.84× / 7.4%</td><td> $1 . 5 4 \times / 6 . 2 \%$ </td></tr><tr><td>Qwen3-0.6B</td><td> $0 . 1 5 \times / 0 . 8 \%$ </td><td> $1 . 1 1 \times / 6 . 0 \%$ </td><td>2.00× / 10.8%</td><td> $1 . 1 3 \times / 6 . 1 \%$ </td></tr></table>

## F.6 End-to-end QA under official protocols

We evaluate how the retrieval gains carry through to answer quality, following each benchmark’s official protocol: LongMemEval’s reading prompt and per-question-type judge prompts, LoCoMo’s QA prompt with normalized-token F1, and ATM-Bench’s official evaluator. The top-10 memories from Base and from the seed-42 Axis-LoRA run are given to the same fixed reader (Qwen3.5-9B) and judge (Qwen3.5-27B) in both conditions, so the delta isolates retrieval. ATM-Hard is omitted because each of its 31 questions carries a ${ \sim } 3$ pp weight on the split score, larger than the QA effects of interest. Retrieval on ATM-Hard, in contrast, is graded over ∼6 gold evidences per query and is reported with query-level confidence intervals (Appendix F.2).

The QA delta (Table 29) matches the sign of the retrieval delta in 8 of 9 cells, including the two retrieval regressions (Qwen3/ATM-Normal, bge/LoCoMo): retrieval gains and dips both propagate to answer quality. The coupling is also visible per query: on ATM-Normal, questions whose gold memory is retrieved in the top-10 are answered correctly 53.7% of the time versus 5.0% when it is missed. The magnitudes are compressed relative to the retrieval deltas because roughly 90% of queries keep the same top-10 (Appendix F.3).

Table 29: End-to-end QA delta (Axis-LoRA − Base, pp) under each benchmark’s official metric (single seed).
<table><tr><td></td><td>LME (judge acc.)</td><td>ATM-N (QS)</td><td>LoCoMo (F1)</td></tr><tr><td>MiniLM</td><td>+2.80</td><td>+0.60</td><td>+0.90</td></tr><tr><td>bge-base</td><td>+2.00</td><td>-0.13</td><td>-0.31</td></tr><tr><td>Qwen3-0.6B</td><td>+2.60</td><td>-1.29</td><td>+0.79</td></tr></table>