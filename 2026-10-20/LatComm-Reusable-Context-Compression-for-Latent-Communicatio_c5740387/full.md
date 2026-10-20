# LatComm: Reusable Context Compression for Latent Communication between Large Language Models

Huawei Lin<sup>1</sup>, Jinting Liu<sup>1</sup>, Tan Yu<sup>2</sup>, Weijie Zhao<sup>1</sup>

<sup>1</sup>Rochester Institute of Technology, <sup>2</sup>NVIDIA

## Abstract

Systems that combine several large language models (LLMs) often use one context for many questions. When the context is passed between models as text, the receiving model processes all of its tokens again for every question. Avoiding this cost requires a context representation that is compact, reusable across questions, and readable by a different LLM. Existing compression and latent communication methods each provide only some of these properties. We introduce LatComm, a Latent Communication interface that provides all three by separating encoding from reading: the context is encoded once without the question, and the receiver selects what to read for each question. A frozen context encoder LLM compresses the context into a fixed-size packet and sends it to a different frozen receiver LLM, which reads the packet through cross-attention, with gates computed from its prompt. On 12 multiplechoice question-answering datasets, LatComm improves average accuracy over single-vector baselines by up to 10.2 points. On QuALITY, it reduces the receiver’s input by 57× and answers the questions of an article 3× faster.

## 1 Introduction

Motivation. Communication between large language models (LLMs) enables models to contribute different roles to a shared task, such as processing documents and answering questions (Wu et al., 2023; Hong et al., 2024; Du et al., 2024; Wang et al., 2025; Lin et al., 2025, 2026). In contextdependent workflows (Lewis et al., 2020; Yao et al., 2023), the same context is often used many times: one document can support many questions, and one observation can inform later decisions. When the context is passed as text, the model that receives it, called the receiver, processes all of its tokens again for every question (Figure 1a). Shared contexts therefore lead to repeated computation across models and questions (Gim et al., 2024; Liu et al.,

![](images/41ad3588b8a05f4132e0b4e121f3efc8395016dd7b9d2456c3fb73c0664a146b.jpg)  
Figure 1: Text passing vs. latent passing. (a) The receiver reads the context again for every question. (b) The context is encoded once and reused. (c, d) Measured cost on QuALITY.

2024b,a). For example, in the reading comprehension dataset QuALITY (Pang et al., 2022), each article comes with 18.1 questions on average. The receiver reads 5,915 tokens for each question, and 98% of them are the article (Figure 1d).

Latent passing. Latent passing avoids this repetition (Figure 1b). An encoder reads the context once and turns it into vectors, called latents. The latents are stored and reused, so the receiver reads only the question. To replace the text, the latents need three properties. They must be reusable: they are computed once, before any question is known. They must be compact: their size does not grow with the context length, so the cost of storing and reading them stays fixed. They must be readable by a different LLM, because systems often combine models of different sizes (Li et al., 2025; Belcak et al., 2025) and from different families (Chen et al., 2024; Ong et al., 2025; Wang et al., 2025). These properties are hard to combine. Latents built without the question and with a fixed size cannot keep only what one question needs, so the receiver must select the relevant information when it reads.

<table><tr><td>Method</td><td>Query- independent</td><td>Fixed-budget compression</td><td>Cross-LLM communication</td><td>Bridge-only training</td></tr><tr><td>LLMLingua-2 (Pan et al., 2024)</td><td>√</td><td>√</td><td>x</td><td>X</td></tr><tr><td>AutoCompressor (Chevalier et al., 2023)</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>ICAE (Ge et al., 2024)</td><td>√</td><td>√</td><td>x</td><td>X</td></tr><tr><td>xRAG (Cheng et al., 2024)</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>CEPE (Yen et al., 2024)</td><td>√</td><td>X</td><td>x</td><td>x</td></tr><tr><td>DroidSpeak (Liu et al., 2024a)</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>C2C (Fu et al., 2026)</td><td>x</td><td>X</td><td>√</td><td>√</td></tr><tr><td>LatentMAS (Zou et al., 2025)</td><td>x</td><td>X</td><td>X</td><td>x</td></tr><tr><td>LatComm (ours)</td><td>√</td><td>L</td><td>L</td><td>J</td></tr></table>

Table 1: Comparison of context compression and latent communication. Query-independent (reusable): the representation is produced without the question. Fixed-budget compression (compact): its size does not depend on the input length. Cross-LLM communication: latent representations pass from one LLM to a different LLM. Bridge-only training: only an interface between two frozen LLMs is trained.

Existing approaches. Existing methods provide only some of these properties. A receiver can store its key–value cache to avoid re-reading (Gim et al., 2024; Liu et al., 2024b), but this cache grows with the context length and can be used only by the model that produced it. Context compression methods shorten what a model reads (Pan et al., 2024; Mu et al., 2023; Chevalier et al., 2023). ICAE compresses a context into fixed memory slots without the question, but it adapts the LLM with LoRA, and the slots are read by the same LLM (Ge et al., 2024). xRAG keeps both models frozen and stores one vector per document (Cheng et al., 2024). This vector is the output of a pretrained retrieval encoder, so the amount of context that it carries is fixed by that encoder. A larger representation would need a larger projector, and its trainable parameters would grow with the size of the representation. Latent communication methods pass representations between models in place of text (Ramesh and Li, 2025; Du et al., 2026; Yang and Eshraghian, 2025; Shi et al., 2026). C2C transfers representations between different frozen LLMs, but both models read the same input, so the transfer depends on the question and grows with the input length (Fu et al., 2026). Table 1 compares representative methods on four properties, and only LatComm has all four.

Approach and results. We introduce LatComm, a latent communication interface that separates encoding from reading (Figure 2). In the encoding step, a frozen context encoder LLM reads the context without the question, and learned context compressors map its token states into a fixed number of low-dimensional slots. These slots form a compressed packet, which is computed once, stored, and sent to the receiver for every question. Because the compressors read all token states of the encoder through cross-attention, a larger packet needs only more slot queries rather than a larger projector. In the reading step, a frozen receiver LLM reads the packet through cross-attention, and gates computed from its prompt control the contribution of each slot, so that different questions can use different parts of the same packet. Both LLMs stay frozen throughout, and only the compressors, readers, and gates are trained. The interface can therefore connect the same model, models of different sizes, or models from different families. On 12 question-answering datasets, LatComm improves average accuracy over single-vector baselines by up to 10.2 points. On QuALITY, it reduces the receiver’s input by 57× and answers all questions of an article 3× faster (Figure 1c, d).

Contributions. Our contributions are as follows:

• We formulate context communication between LLMs as two steps, encoding without the question and reading with the question, with a packet of fixed size between them.

• We introduce LatComm, an interface between two frozen LLMs with context compressors, crossattention readers, and per-layer, per-slot gates. It is trained in two stages, so that packets stored after the first stage remain valid.

• We show that LatComm is more accurate than single-vector baselines, answers questions three times faster than text passing, and connects six LLM pairs of different sizes and families.

• We release a PyTorch implementation with training and evaluation configurations.<sup>1</sup>

![](images/acbdddad8fb82dd767fb39d909f9e1d5e08893cd283169c9f9fbef98ebd75cc7.jpg)  
Figure 2: LatComm overview. (a) The encoder reads the context; the receiver reads a task prompt. LatComm connects selected layers and adds context information after each paired receiver block. Both LLMs remain frozen. (b) For one layer pair, a context compressor produces reusable slots Z . The gates compare a summary of the receiver’s prompt states with the compressed slots. The reader combines the gated slot values and adds the result to the receiver state.

## 2 Related Work

Context compression. Context compression reduces the input that an LLM reads. LLMLingua-2 keeps a subset of the tokens (Pan et al., 2024), so its output is still text. Gist tokens, AutoCompressor, and ICAE train an LLM to encode a prompt or a context into a few vectors that the same LLM reads (Mu et al., 2023; Chevalier et al., 2023; Ge et al., 2024). xRAG maps one retrieval embedding per document into a frozen LLM (Cheng et al., 2024). CEPE and Flamingo feed a separately encoded input to a frozen language model through inserted cross-attention layers (Yen et al., 2024; Alayrac et al., 2022). CEPE trains a small encoder for its decoder and keeps one vector per context token, and the gates of Flamingo are learned constants. LatComm compresses the hidden states of one LLM into a fixed number of slots, which a different LLM reads through gates that change with its prompt.

Latent communication. Latent communication passes internal representations between models in place of text, such as activations, hidden states, translated vectors, or selected key–value (KV) pairs (Ramesh and Li, 2025; Du et al., 2026; Yang and Eshraghian, 2025; Shi et al., 2026). DroidSpeak reuses KV caches across LLMs with the same architecture (Liu et al., 2024a), and LatentMAS passes the KV caches of one agent to the next without training (Zou et al., 2025). C2C trains cache fusers between different frozen LLMs (Fu et al., 2026). It transfers per-token states of an input that the two models both read, so the transfer grows with the input length. LatComm instead sends a packet with several slots per layer that is built without the question, and each receiver token reads it through gates computed from the prompt.

## 3 LatComm: Latent Communication

Consider a context x, such as a document, and a prompt r that contains a question about x, the task instructions, and any answer choices. With text passing, the receiver LLM reads x followed by r and generates the answer y, so it processes x again for every prompt (Figure 1a). With LatComm, the receiver reads only r (Figure 1b). A second LLM, the context encoder LLM, reads x once and sends a compressedpacket Z(x), a fixed number of learned vectors, and the receiver generates y from r and Z(x). The packet is the receiver’s only source of context information, and both LLMs stay frozen.

The packet passes between selected layers of the two models (Figure 2a). At each of these layer pairs, a compressor turns the encoder’s token states, the hidden states that the layer computes for each token, into a few slots. A reader with gates then adds the information that the current prompt needs to the receiver’s states (Figure 2b). Only the compressors, readers, and gates are trained, in two stages.

![](images/55795cd3fc3770a23925f1f7c6bd9691cae4674f92d2efb5ca513b9c08cf38b4.jpg)  
Figure 3: Two-stage training of LatComm. (a) In Stage 1, the receiver gets an instruction to reproduce the context x and can read it only from the packet $\mathcal { Z } _ { \phi } ( x )$ . The reconstruction loss $\mathcal { L } _ { \mathrm { r e c } }$ trains the compressors, readers, and gates. (b) In Stage 2, the receiver gets the question with its answer choices, and the task loss $\mathcal { L } _ { \mathrm { t a s k } }$ on the answer y trains only the readers and gates. The compressors are fixed, so the packet of each training context is computed once and stored. Both LLMs stay frozen in both stages.

## 3.1 Encoding a Compressed Packet

We use P connections, called layer pairs. Pair i connects encoder layer $a _ { i }$ to receiver layer $b _ { i }$ . Its context compressor maps the encoder’s token states $H _ { a _ { i } } ^ { E } ( x ) \in \mathbb { R } ^ { n _ { E } \times d _ { E } }$ into $S _ { i }$ slots of dimension $d ,$ where $n _ { E }$ is the number of context tokens and $d _ { E }$ is the encoder’s hidden dimension. With learned slot queries $U _ { i } \in \mathbb { R } ^ { S _ { i } \times d }$ , compression is

$$
\begin{array} { r l } & { X _ { i } = H _ { a _ { i } } ^ { E } ( x ) W _ { i } ^ { E } , } \\ & { C _ { i } = \mathrm { M H A } _ { i } ( U _ { i } , X _ { i } , X _ { i } ) , } \\ & { B _ { i } = \mathrm { L N } _ { i } ^ { \mathrm { a t t n } } ( U _ { i } + C _ { i } ) , } \\ & { Z _ { i } ( x ) = \mathrm { L N } _ { i } ^ { \mathrm { f i n } } \big ( B _ { i } + \mathrm { F F N } _ { i } ( B _ { i } ) \big ) . } \end{array}\tag{1}
$$

The projection $W _ { i } ^ { E }$ maps context states to width d. Multi-head cross-attention (MHA) lets the slot queries collect information from these states through parallel weighted reads. Layer normalization (LN) normalizes features, and a feed-forward network (FFN) transforms the result. Each row of $Z _ { i } ( \boldsymbol { x } ) \ \in \ \mathbb { R } ^ { S _ { i } \times d }$ is one slot. Each pair has its own compressor, and the packet $\mathcal { Z } ( x ) \ =$ $( Z _ { 1 } ( x ) , \ldots , Z _ { P } ( x ) )$ collects the slots of all pairs.

The compressors read hidden states rather than key–value (KV) caches. A hidden state is one vector per token, whereas a KV cache holds keys and values for every attention head and is tied to the token positions of its input, so methods that transfer caches align the tokens of the two models (Fu et al., 2026). Slots pooled from hidden states have no token positions, and the receiver can read them without reading the context.

The encoder and the compressors see only the context, without the question or the answer choices, so the same context always produces the same packet, and the receiver can reuse it for different prompts. The packet stores d $\textstyle \sum _ { i } S _ { i }$ numbers, which depends on the number of slots and their width but not on the number of context tokens. Because the slots come from learned queries, a larger packet needs only more queries, and the size of the compressor stays the same. The compressors are fixed after the first training stage, so each context is encoded once and its packet can be stored.

## 3.2 Reading with Per-Layer, Per-Slot Gates

For each prompt, the receiver runs its usual layers, and after layer $b _ { i }$ a trainable reader gives it access to the slots $Z _ { i }$ . Let $H _ { i } \in \mathbb { R } ^ { n _ { T } \times d _ { T } }$ be the receiver states after layer $b _ { i } ,$ where $n _ { T }$ is the number of receiver tokens and $d _ { T }$ is the receiver’s hidden dimension. The reader uses cross-attention, so each token assigns its own weights to the slots. These weights sum to one over the slots, so attention alone can shift weight between slots but cannot reduce the total contribution of the packet. A gate $g _ { i , s } \in ( 0 , 1 )$ therefore sets the strength of slot s at pair i. A learned pooling module, shared across pairs, summarizes the receiver states at the prompt positions into a vector $p _ { i }$ , and the gate compares this summary with the slot $z _ { i , s } = Z _ { i } ( x ) [ s , : ]$

$$
g _ { i , s } = \sigma \left( \frac { \langle p _ { i } W _ { g } ^ { Q } , z _ { i , s } W _ { g } ^ { K } \rangle } { \sqrt d } + \beta _ { i , s } \right) .\tag{2}
$$

Here $\sigma$ is the sigmoid, the projections $W _ { g } ^ { Q }$ and $W _ { g } ^ { K }$ are shared across pairs, and every layer–slot pair has its own bias $\beta _ { i , s }$ . At later layers, the prompt states already contain context information from earlier readers, so $p _ { i }$ and the gates depend on both x and $r .$

The reader projects the normalized receiver states into queries $Q _ { i }$ and the normalized slots into keys $K _ { i }$ and values $V _ { i }$ . It then scales the values with the gates and adds the result to the receiver states. For one attention head,

$$
\begin{array} { r l } & { A _ { i } = \mathrm { s o f t m a x } \big ( Q _ { i } K _ { i } ^ { \top } / \sqrt { d } \big ) , } \\ & { H _ { i } ^ { \prime } = H _ { i } + A _ { i } G _ { i } V _ { i } W _ { i } ^ { O } , } \end{array}\tag{3}
$$

where $G _ { i } = \mathrm { d i a g } ( g _ { i , 1 } , . . . , g _ { i , S _ { i } } )$ . The gates scale the values and leave the attention weights unchanged, and $H _ { i } ^ { \prime }$ continues through the rest of the receiver. We call this dynamic per-layer, per-slot value gating: every layer pair and slot has its own gate, and its value changes with the prompt. A gate is computed when the receiver processes the prompt, is shared by all receiver tokens, and is reused while the answer is generated. The pooling module, the multi-head form of the reader, and the initialization are specified in Appendix A.

## 3.3 Training

LatComm is trained in two stages (Figure 3). Stage 1, context reconstruction, teaches the interface to carry context information, and Stage 2, task supervision, teaches the receiver side to use it for a task. Let $\theta = ( \phi , \psi , \eta )$ be the parameters of the compressors, readers, and gates. Both stages minimize the cross-entropy of the response tokens under teacher forcing. For a minibatch B, let example $j$ have the prompt $r _ { j }$ , the context $x _ { j }$ , and the response $y _ { j }$ with supervised positions $\mathcal { R } _ { j }$ , and let $\ell _ { j , t } = -$ log $P _ { \theta } ( y _ { j , t } \mid r _ { j } , \mathcal { Z } _ { \phi } ( x _ { j } ) , y _ { j , < t } )$ be the loss of response token t. The minibatch loss is

$$
\mathcal { L } ( \boldsymbol { \theta } ; \mathcal { B } ) = \frac { \sum _ { j \in \mathcal { B } } \sum _ { t \in \mathcal { R } _ { j } } \ell _ { j , t } } { \sum _ { j \in \mathcal { B } } | \mathcal { R } _ { j } | } .\tag{4}
$$

Context reconstruction. Stage 1 trains all of $\theta .$ The receiver gets an instruction to reproduce the context, and the target is the context itself, tokenized by the receiver and followed by an endof-sequence token. The receiver does not see the context in its prompt, so it can reproduce the context only from the packet. Predicting every context token trains the packet to keep information that different questions may need.

Task supervision. Stage 2 starts from the Stage 1 weights, keeps the compressors $\phi$ fixed, and trains only the readers $\psi$ and the gates η on the task data.

<table><tr><td>Method</td><td>Compute per question</td><td>One-time compute</td><td>Stored size</td><td>Cross- LLM</td></tr><tr><td>Text passing</td><td> $L _ { T } ( n _ { E } + n _ { T } ) ^ { 2 }$ </td><td></td><td> $n _ { E }$ </td><td>√</td></tr><tr><td>KV cache</td><td> $L _ { T } n _ { T } ( n _ { E } + n _ { T } )$ </td><td> $L _ { T } n _ { E } ^ { 2 }$ </td><td> $L _ { T } n _ { E } d _ { T }$ </td><td>X</td></tr><tr><td>LatComm</td><td> ${ \cal L } _ { T } n _ { T } ^ { 2 } + { \cal P } n _ { T } { \cal S }$ </td><td> $L _ { E } n _ { E } ^ { 2 }$ </td><td> $P S d$ </td><td>√</td></tr></table>

Table 2: Orders of growth for three ways to reuse a context of n tokens for questions with $n _ { T ^ { - } } \mathrm { t o k e n }$ prompts. $L _ { E }$ and $L _ { T }$ are the numbers of encoder and receiver layers. LatComm uses $P$ layer pairs with S slots of width $d .$ The compute columns treat hidden sizes as constants and assume full self-attention in every layer.

For multiple-choice questions, the target is the correct option letter. Because the compressors are fixed, the packet of each training context is computed once and stored, and the context encoder LLM is not run again in this stage. For the same reason, packets that were stored before task training remain valid after it.

## 3.4 Complexity

Consider a context of $n _ { E }$ tokens that is used for q questions with prompts of $n _ { T }$ tokens, and let $c _ { E }$ and $c _ { T }$ be the cost per token of the context encoder and of the receiver. Reading the context as text for every question costs $C _ { \mathrm { t e x t } }$ , and encoding it once costs $C _ { \mathrm { p a c k e t } }$

$$
\begin{array} { l } { { C _ { \mathrm { t e x t } } = q \left( n _ { E } + n _ { T } \right) c _ { T } , } } \\ { { C _ { \mathrm { p a c k e t } } = n _ { E } c _ { E } + q n _ { T } c _ { T } . } } \end{array}\tag{5}
$$

Encoding once therefore costs less when $q \mathrm { ~ ‰ ~ }$ $c _ { E } / c _ { T }$ . Table 2 compares the growth rates with text passing and with a stored KV cache of the receiver. The receiver’s compute for each question grows quadratically with the context length for text and linearly for a KV cache, because each prompt token still attends to all $n _ { E }$ context positions. For LatComm it does not depend on $n _ { E } .$ because each prompt token attends to S slots at each of the $P$ layer pairs. The readers and gates add $O ( n _ { T } d _ { T } d + n _ { T } S _ { i } d )$ work per layer pair, which Equation 5 omits. The stored size grows with $n _ { E }$ for the cache and is fixed for the packet: with 24 layer pairs, 32 slots per pair, $d = 2 5 6$ , and 16-bit values, a packet takes 384 KiB.

## 4 Experiments

The experiments address three research questions. RQ1: Can a context be encoded once, without the question, and reused for different questions? RQ2: How much useful information does a packet keep under a fixed size budget? RQ3: Can two different LLMs communicate through the packet, across model sizes and model families?

<table><tr><td rowspan="2">Method</td><td rowspan="2">Encoder</td><td colspan="10">Task datasets (%)</td><td colspan="4">Held-out datasets (%)</td></tr><tr><td>CLOTH</td><td>Cosmos HSwag</td><td></td><td>LogiQA MCScr.</td><td></td><td>MuTual OBQA</td><td>QuAL.</td><td>ReClor</td><td>SciQ SIQA</td><td>SWAG Avg</td><td></td><td>RACE DREAM QuAIL</td><td></td><td>Avg</td></tr><tr><td>Full text</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Raw-text</td><td></td><td>45.3</td><td>40.8</td><td>37.0</td><td>31.0 82.3</td><td>48.0</td><td>68.0</td><td>44.9</td><td>38.6</td><td>97.5 48.2</td><td>35.6 51.43</td><td>66.1</td><td>70.9</td><td>58.2</td><td>65.06</td></tr><tr><td colspan="10">No context</td><td colspan="7"></td></tr><tr><td>Query-only</td><td></td><td>25.0</td><td>44.8</td><td>28.9</td><td>26.0</td><td>62.6</td><td>32.3 53.6</td><td>32.2</td><td>27.8</td><td>77.1 37.7</td><td>27.3</td><td>39.60</td><td>39.1</td><td>36.2</td><td>34.0</td><td>36.41</td></tr><tr><td>Constant packet</td><td></td><td>44.2</td><td>63.2</td><td>68.9</td><td>37.5</td><td>75.6</td><td>71.9</td><td>68.8 39.5</td><td>41.8</td><td>88.1</td><td>57.6 65.8</td><td>60.25</td><td>45.5</td><td>41.2</td><td>43.5</td><td>43.37</td></tr><tr><td colspan="10">Compressed context</td><td colspan="7"></td></tr><tr><td>xRAG</td><td>SFR-Mistral</td><td>41.2</td><td>55.9</td><td>55.1</td><td>33.6</td><td>76.0</td><td>52.0</td><td>71.2</td><td>40.0</td><td>36.4</td><td>89.4 57.9 61.6</td><td>55.87</td><td>47.4</td><td>48.8</td><td>43.9</td><td>46.70</td></tr><tr><td>xRAG</td><td>Qwen3.5-4B</td><td>38.5</td><td>53.1</td><td>56.1</td><td>36.4</td><td>73.7</td><td>58.6</td><td>69.6 37.7</td><td>35.4</td><td>88.1</td><td>54.8 61.2</td><td>55.27</td><td>44.8</td><td>44.8</td><td>44.6</td><td>44.71</td></tr><tr><td>Embedding</td><td>F2LLM-v2</td><td>45.1</td><td>58.6</td><td>68.3</td><td>35.2</td><td>78.7</td><td>72.3</td><td>74.4</td><td>41.1</td><td>39.4 87.1</td><td>61.9 68.6</td><td>60.90</td><td>44.2</td><td>44.9</td><td>36.1</td><td>41.72</td></tr><tr><td>Embedding</td><td>BGE-Gemma2</td><td>46.2</td><td>58.4</td><td>70.5</td><td>36.1</td><td>78.6</td><td>71.7</td><td>73.2</td><td>43.7</td><td>40.4 89.8</td><td>64.9 70.2</td><td>61.98</td><td>46.1</td><td>46.6</td><td>41.7</td><td>44.83</td></tr><tr><td>LatComm</td><td>Qwen3.5-4B</td><td>52.1</td><td>61.9</td><td>73.5</td><td>38.6</td><td>80.9</td><td>74.0</td><td>79.2</td><td>43.1</td><td>49.8 92.4</td><td>66.5 73.1</td><td>65.43</td><td>52.2</td><td>52.3</td><td>44.8</td><td>49.76</td></tr></table>

Table 3: Main results with the Qwen3.5-0.8B receiver: accuracy on each dataset. Raw-text and Query-only are the untrained receiver with and without the context in its prompt, and Constant packet trains the readers and gates with one fixed packet. Avg is Task Avg or Held-out Avg. Bold marks the best compressed-context method per column.

<table><tr><td>Context encoder</td><td>Receiver</td><td>P</td><td>S</td><td>d</td><td>Packet</td></tr><tr><td colspan="6">Same model</td></tr><tr><td>Qwen3.5-0.8B</td><td>Qwen3.5-0.8B</td><td>24</td><td>32</td><td>256</td><td>384 KiB</td></tr><tr><td>Larger encoder</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-4B Qwen3.5-9B</td><td>Qwen3.5-0.8B Qwen3.5-0.8B</td><td>24 24</td><td>32 32</td><td>256 256</td><td>384 KiB 384 KiB</td></tr><tr><td>Cross-family</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-0.8B</td><td>Gemma-3-1B-pt</td><td>24</td><td>32</td><td>256</td><td>384 KiB</td></tr><tr><td>Qwen3.5-9B</td><td>Gemma-3-1B-pt</td><td>26</td><td>32</td><td>256</td><td>416 KiB</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Granite-4.1-8B</td><td>Llama-3.2-1B</td><td>16</td><td>32</td><td>256</td><td>256 KiB</td></tr></table>

Table 4: Model pairs with the default packet. P is the number of layer pairs, S the number of slots per pair, and d the slot width. The packet stores $P \times S \times d$ values in bfloat16, which takes 2PSd bytes.

## 4.1 Setup

Datasets. Stage 1 uses 2,000,000 English passages from WikiText-103 (Merity et al., 2017) and FineWeb-Edu (Penedo et al., 2024). Stage 2 uses 12 multiple-choice datasets in which each question comes with a context: CLOTH (Xie et al., 2018), Cosmos QA (Huang et al., 2019), HellaSwag (Zellers et al., 2019), LogiQA (Liu et al., 2020), MCScript2.0 (Ostermann et al., 2019), Mu-Tual (Cui et al., 2020), OpenBookQA (Mihaylov et al., 2018), QuALITY (Pang et al., 2022), Re-Clor (Yu et al., 2020), SciQ (Welbl et al., 2017), SocialIQA (Sap et al., 2019), and SWAG (Zellers et al., 2018). We also test on three held-out datasets not used in training: RACE (Lai et al., 2017), DREAM (Sun et al., 2019), and QuAIL (Rogers et al., 2020) (sizes and splits in Appendix B). The prediction is the valid option letter with the highest logit at the first response position. We report the accuracy on each dataset and two unweighted means: Task Avg over the 12 task datasets and Held-out Avg over the held-out datasets.

Model pairs. Table 4 lists the six model pairs. The main receiver is Qwen3.5-0.8B (Qwen Team, 2026), with the same model or a larger Qwen3.5 model (4B or 9B) as the context encoder. Two more receivers cross model families: Gemma-3-1Bpt (Gemma Team, 2025) with Qwen3.5 encoders, and Llama-3.2-1B (Meta, 2024) with the Granite-4.1-8B base model (IBM Granite Team, 2026) as the encoder. Each reader is paired with the encoder layer at the same relative depth, and the default packet has 32 slots per pair of width d = 256.

Baselines. We group the compared methods by what the receiver gets. Full text: Raw-text gives the frozen receiver the original context before the prompt. It is not trained on the task data, so we report it as a reference. No context: Query-only gives the frozen receiver only the prompt, and Constant packet trains the same readers and gates as LatComm but gives every question the same packet, so it measures how much accuracy comes from task training alone. Compressed context: xRAG (Cheng et al., 2024) maps one context vector to one soft token, where the vector comes from SFR-Embedding-Mistral (Salesforce AI Research, 2024) or from the last-token state of Qwen3.5-4B, LatComm’s encoder. Embedding replaces the packet with one vector from a text embedding model, F2LLM-v2- 14B (Zhang et al., 2026) or BGE-Multilingual-Gemma2 (Beijing Academy of Artificial Intelligence, 2024), which the same readers and gates read. Appendix C describes each baseline. All trained methods share the task data, loss, batch size, epochs, and learning-rate schedule of Table 9.

<table><tr><td>Method</td><td>Encoder</td><td colspan="10">Task datasets (%)</td><td colspan="5">Held-out datasets (%)</td></tr><tr><td></td><td></td><td>CLOTH Cosmos HSwag LogiQA MCScr. MuTual OBQA QuAL. ReClor</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>SciQ</td><td>SIQA SWAG</td><td>Avg</td><td></td><td>RACE DREAM QuAIL</td><td></td><td>Avg</td></tr><tr><td>Full text</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Raw-text</td><td></td><td>25.0</td><td>25.4</td><td>25.0</td><td>22.0</td><td>50.9</td><td>23.6</td><td>28.0</td><td>25.6</td><td>24.4 24.8 33.5</td><td>24.6</td><td>27.73</td><td>25.9</td><td>34.7</td><td>25.2</td><td>28.63</td></tr><tr><td>No context</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Query-only</td><td></td><td>25.4</td><td>25.5</td><td>25.7</td><td>25.8</td><td>48.9</td><td>23.8</td><td>26.4</td><td>25.9</td><td>25.8 23.5</td><td>33.7 25.2</td><td>27.98</td><td>26.4</td><td>34.6</td><td>25.2</td><td>28.74</td></tr><tr><td>Constant packet</td><td></td><td>38.2</td><td>59.3</td><td>67.9</td><td>28.4</td><td>71.8</td><td>59.9</td><td>56.8</td><td>35.5</td><td>28.2 82.0</td><td>54.8 65.7</td><td>54.04</td><td>39.0</td><td>41.6</td><td>43.3</td><td>41.30</td></tr><tr><td>Compressed context</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Embedding</td><td>F2LLM-v2</td><td>37.4</td><td>57.8</td><td>64.0</td><td>29.6</td><td>71.8</td><td>59.4</td><td>57.4</td><td>33.1</td><td>30.4 80.4</td><td>57.1 64.9</td><td>53.62</td><td>38.8</td><td>40.5</td><td>37.7</td><td>38.98</td></tr><tr><td>Embedding</td><td>BGE-Gemma2</td><td>37.8</td><td>60.1</td><td>68.4</td><td>30.0</td><td>72.1</td><td>58.5</td><td>60.8</td><td>36.5</td><td>33.6 82.0</td><td>58.8 66.9</td><td>55.46</td><td>41.2</td><td>41.9</td><td>38.6</td><td>40.59</td></tr><tr><td>LatComm</td><td>Qwen3.5-9B</td><td>46.3</td><td>58.9</td><td>70.1</td><td>32.1</td><td>78.8</td><td>64.1</td><td>68.4</td><td>31.8</td><td>36.4</td><td>85.6 62.5 73.0</td><td>59.00</td><td>43.2</td><td>49.9</td><td>46.8</td><td>46.63</td></tr></table>

Table 5: Results with the Gemma-3-1B-pt receiver: accuracy on each dataset. Raw-text and Query-only are the untrained receiver with and without the context in its prompt, and both are near chance for this receiver. Avg is Task Avg or Held-out Avg. Bold marks the best compressed-context method per column.

## 4.2 Main Results

Table 3 compares all methods with the Qwen3.5- 0.8B receiver on the 12 task datasets and on the three held-out datasets. LatComm is the best compressed-context method on both averages. On the task datasets, it reaches 65.43 Task Avg, against 61.98 for the best baseline, and it has the highest accuracy among the compressed-context methods on 11 of the 12 datasets. On the held-out datasets, which are not used in training, it reaches 49.76 Held-out Avg, against 46.70 for the best baseline, and it has the highest accuracy on all three datasets. LatComm is also above the Constant packet by 5.2 points on Task Avg and by 6.4 points on Held-out Avg. The Constant packet uses the same trained readers and gates but carries no context information, so this gain shows that the packet gives the receiver useful information about the context.

Raw-text is not trained on the task data. Every trained method is above Raw-text on the task datasets and below it on the held-out datasets (41.72–49.76 against 65.06 Held-out Avg), and this includes the Constant packet. The modules trained on the 12 task datasets therefore transfer only in part to new datasets. This holds for every trained method, and LatComm is the most accurate.

## 4.3 Cross-Family Receiver

Table 5 repeats the comparison with a receiver from another model family, Gemma-3-1B-pt, and with the Qwen3.5-9B context encoder. Without training, this receiver is close to chance level with or without the context (27.73 and 27.98 Task Avg). LatComm reaches 59.00 Task Avg, against 55.46 for the best baseline and 54.04 for the Constant packet. On the held-out datasets, LatComm reaches 46.63 Heldout Avg, against 40.59 for the best baseline and 41.30 for the Constant packet. It has the highest accuracy among the compressed-context methods on 10 of the 12 task datasets and on all three heldout datasets. The interface therefore also works when the context encoder and the receiver come from different model families.

<table><tr><td>Encoder</td><td>Receiver</td><td>Constant packet Embedding</td><td></td><td>LatComm</td></tr><tr><td>Same model</td><td></td><td></td><td>61.98</td><td>64.91</td></tr><tr><td>Qwen3.5-0.8B</td><td>3 Qwen3.5-0.8B</td><td>60.25</td><td></td><td></td></tr><tr><td>Larger encoder</td><td></td><td></td><td></td><td>65.43</td></tr><tr><td>Qwen3.5-4B</td><td>Qwen3.5-0.8B</td><td>60.25</td><td>61.98</td><td></td></tr><tr><td>Qwen3.5-9B</td><td>Qwen3.5-0.8B</td><td>60.25</td><td>61.98</td><td>65.80</td></tr><tr><td>Cross-family</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-0.8B</td><td>Gemma-3-1B-pt</td><td>54.04</td><td>55.46</td><td>59.21</td></tr><tr><td>Qwen3.5-9B</td><td>Gemma-3-1B-pt</td><td>54.04</td><td>55.46</td><td>59.00</td></tr><tr><td>Granite-4.1-8B Llama-3.2-1B</td><td></td><td>60.62</td><td>61.42</td><td>64.08</td></tr></table>

Table 6: Task Avg (%) of six model pairs and two trained references with the same receiver. Embedding uses BGE-Multilingual-Gemma2. Each receiver’s Constant packet was trained with its 4B, 9B, or Granite pair.

The two Embedding baselines are below the Constant packet on Held-out Avg. This difference comes from QuAIL. Every QuAIL question has a “not enough information” option, which is correct for 306 of the 2,720 questions. The Embedding baselines choose it for 18.5% and 25.0% of the questions, against 7.4% for the Constant packet and 7.5% for LatComm. On RACE and DREAM, the Embedding baselines are within 2.3 points of the Constant packet.

## 4.4 Cross-LLM Communication

For RQ3, Table 6 compares LatComm with two trained references for six pairs of LLMs. In every pair, the receiver does not read the context, so its context information comes only from the packet that the context encoder LLM sends. With the Qwen3.5-0.8B receiver, Task Avg is 64.91 when the same model is the context encoder, 65.43 with the 4B encoder, and 65.80 with the 9B encoder. The three encoders are within 0.9 points of each other, so the interface works with a small encoder, and one model can serve as both encoder and receiver. Across model families, the Gemma-3- 1B-pt receiver reaches 59.00–59.21 with Qwen3.5 encoders, and the Llama-3.2-1B receiver reaches 64.08 with the Granite-4.1-8B encoder. In all six pairs, LatComm is 3.5 to 5.6 points above the Constant packet and 2.7 to 3.8 points above the Embedding baseline, both trained with the same receiver. The packet thus passes context information between LLMs of different sizes and families.

<table><tr><td>Quantity</td><td>Raw-text</td><td>LatComm</td></tr><tr><td>Per question</td><td></td><td></td></tr><tr><td>Receiver input tokens</td><td>5,915</td><td>103</td></tr><tr><td>Receiver time (ms)</td><td>511</td><td>131</td></tr><tr><td>Receiver peak memory (GiB)</td><td>0.87</td><td>0.06</td></tr><tr><td>Per context</td><td></td><td></td></tr><tr><td>Encoding time (s)</td><td></td><td>0.71</td></tr><tr><td>Total time, 18.1 questions (s)</td><td>9.26</td><td>3.09</td></tr></table>

Table 7: Measured cost on the QuALITY test set (115 articles, 2,086 questions) with the Qwen3.5-4B encoder and the Qwen3.5-0.8B receiver. Times are measured on one H100 GPU at batch size 1. Peak memory is the GPU memory that one call adds to the loaded models.

The receiver’s answers depend on the packet it receives. With the Qwen3.5-4B encoder and the 0.8B receiver, the same readers and gates reach 65.4 Task Avg with the packet of the question’s own context, 45.6 with the packet of another context, and 39.6 without a packet (Table 11).

## 4.5 Reuse Cost

For RQ1, one packet serves every question about its context. In the test set, one context has 18.1 questions on average in QuALITY, 14.2 in CLOTH, and 5.7 in MCScript2.0. Table 7 reports the measured cost on QuALITY. Encoding an article once takes 0.71 s. The receiver then answers a question in 131 ms with the packet, against 511 ms from the full text (Figure 1d). A Query-only pass takes 111 ms, so the readers and gates add only 20 ms per question. For one article with 18.1 questions, the packet path takes 3.09 s in total and the text path takes 9.26 s, so the packet path is 3.0 times faster (Figure 1c). Encoding once costs less from

![](images/4ae7db32224d542af87f66f23781e5672c3a8d0dbdfc3fa2d01288e7a4aea832.jpg)

![](images/0419a65d45c438bfc668cc6a7c396690f52fa413cc10680c74eaa2ddfc7737d2.jpg)  
(b) Encoding  
Figure 4: Reuse cost on QuALITY. (a) Receiver time per question for Raw-text, five LatComm settings, and Query-only. The hatched part of a bar is the time added by the readers and gates. (b) Encoding time per context for three context encoders.

the second question.

The gain grows with the context length (Figure 7). With text passing, the time per question grows from 300 ms for the fifth of the questions with the shortest articles to 651 ms for the fifth with the longest. With a packet it stays near 131 ms, so the receiver is 2.3 to 4.9 times faster. Figure 4a compares the settings of LatComm with Raw-text and Query-only. Every setting takes between 121 and 133 ms per question. The slot count does not change the time, and with 6 layer pairs the readers and gates add 6.2 ms instead of 20.5 ms. Encoding takes 0.54 to 0.73 s with the three encoders (Figure 4b).

For RQ2, the default packet takes 384 KiB per context. Task Avg stays between 65.0 and 65.8 when the slot count changes from 8 to 64, so a 96 KiB packet with 8 slots per pair keeps the accuracy of the default packet. A slot width of 512 raises Task Avg to 66.88. Every packet configuration is above the best single-vector baseline (61.98). Appendix D reports all configurations.

## 5 Conclusion

We presented LatComm, a latent communication interface between frozen LLMs. A context encoder LLM compresses a context once into a fixed-size packet, and a receiver LLM, which can differ from the encoder in size and family, reads it for every question through cross-attention with gates computed from its prompt. LatComm improves average accuracy over single-vector baselines by up to 10.2 points, and on QuALITY it answers the questions of an article 3× faster than text passing. Future work will study packets as messages in multi-agent systems and as entries in agent memory.

## Limitations

Our experiments cover multiple-choice question answering in English, where the answer is a single option letter. We do not evaluate free-form generation, and we do not test the multi-agent and agent-memory uses that motivate reusable packets. The receivers have about one billion parameters and the encoders have at most nine billion, so the behavior of the interface with larger receivers is not known. In addition, only Stage 2 of the default pair is repeated with three seeds, which gives a standard deviation of 0.23 points in Task Avg. Every other configuration uses one seed, so small differences between the configurations of the ablation study should be read with care.

LatComm is trained for one pair of models, so a new context encoder or receiver needs its own compressors, readers, and gates, and its own packets. The interface also reads and writes hidden states, and therefore it does not apply to models that are available only through a text interface. Finally, because the packet has a fixed size, a longer context is compressed more strongly. Stage 1 uses passages of at most 1,024 tokens, and the longest contexts in our evaluation are the QuALITY articles. We do not study how the packet size should grow with the context length.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, and 8 others. 2022. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, NeurIPS, New Orleans, LA.

Beijing Academy of Artificial Intelligence. 2024. BGE-Multilingual-Gemma2: Model card.

Peter Belcak, Greg Heinrich, Shizhe Diao, Yonggan Fu, Xin Dong, Saurav Muralidharan, Yingyan Celine Lin, and Pavlo Molchanov. 2025. Small language models are the future of agentic AI. CoRR, abs/2506.02153.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. Frugalgpt: How to use large language models while reducing cost and improving performance. Trans. Mach. Learn. Res., 2024.

Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. 2024. xrag: Extreme context compression for

retrieval-augmented generation with one token. In Advances in Neural Information Processing Systems, NeurIPS, Vancouver, BC, Canada.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP, pages 3829–3846, Singapore. Association for Computational Linguistics.

Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. 2020. Mutual: A dataset for multi-turn dialogue reasoning. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 1406–1416.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, ICML, volume 235 of Proceedings ofMachine Learning Research, pages 11733–11763, Vienna, Austria.

Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, and Haochao Ying. 2026. Enabling agents to communicate entirely in latent space. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, ACL, pages 27106–27129, San Diego, California.

Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu Wang. 2026. Cache-to-cache: Direct semantic communication between large language models. In The Fourteenth International Conference on Learning Representations, ICLR, Rio de Janeiro, Brazil.

Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations, ICLR, Vienna, Austria.

Gemma Team. 2025. Gemma-3-1B-pt: Model card.

In Gim, Guojun Chen, Seung-Seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. 2024. Prompt cache: Modular attention reuse for low-latency inference. In Proceedings of the Seventh Annual Conference on Machine Learning and Systems, MLSys, Santa Clara, CA.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. Metagpt: Meta programming for A multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, ICLR, Vienna, Austria.

Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2019. Cosmos QA: machine reading

comprehension with contextual commonsense reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP, pages 2391–2401, Hong Kong, China.

IBM Granite Team. 2026. Granite-4.1-8B-Base: Model card.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard H. Hovy. 2017. RACE: large-scale reading comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP, pages 785–794, Copenhagen, Denmark.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, NeurIPS, Virtual.

Senyao Li, Haozhao Wang, Wenchao Xu, Rui Zhang, Song Guo, Jingling Yuan, Xian Zhong, Tianwei Zhang, and Ruixuan Li. 2025. Collaborative inference and learning between edge slms and cloud llms: A survey of algorithms, execution, and open challenges. CoRR, abs/2507.16731.

Huawei Lin, Peng Li, Jie Song, Fuxin Jiang, and Tieying Zhang. 2026. MUSE-Autoskill: Self-evolving agents via skill creation, memory, management, and evaluation. CoRR, abs/2605.27366.

Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, and Ravender Pal Singh. 2025. Agentomni: Test-time multimodal reasoning via model coordination for understanding anything. CoRR, abs/2511.02834.

Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2020. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In Proceedings ofthe Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI, pages 3622–3628.

Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, et al. 2024a. Droidspeak: Kv cache sharing for cross-llm communication and multi-llm serving. arXiv preprint arXiv:2411.02820.

Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, and Junchen Jiang. 2024b. Cachegen: KV cache compression and streaming for fast large language model serving. In Proceedings of the ACM SIGCOMM 2024 Conference, ACM SIGCOMM, pages 38–56, Sydney, NSW, Australia.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In 5th International Conference on Learning Representations, ICLR, Toulon, France.

Meta. 2024. Llama-3.2-1B: Model card.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP, pages 2381–2391, Brussels, Belgium.

Jesse Mu, Xiang Li, and Noah D. Goodman. 2023. Learning to compress prompts with gist tokens. In Advances in Neural Information Processing Systems, NeurIPS, New Orleans, LA.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica. 2025. Routellm: Learning to route llms from preference data. In The Thirteenth International Conference on Learning Representations, ICLR, Singapore.

Simon Ostermann, Michael Roth, and Manfred Pinkal. 2019. Mcscript2.0: A machine comprehension corpus focused on script events and participants. In Proceedings ofthe Eighth Joint Conference on Lexical and Computational Semantics, pages 103–117, Minneapolis, MN.

Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang. 2024. Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. In Findings of the Association for Computational Linguistics, ACL, volume ACL 2024 of Findings ofACL, pages 963–981, Bangkok, Thailand.

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel R. Bowman. 2022. Quality: Question answering with long input texts, yes! In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics, NAACL, pages 5336–5358, Seattle, WA.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin A. Raffel, Leandro von Werra, and Thomas Wolf. 2024. The fineweb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, NeurIPS, Vancouver, BC, Canada.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Vignav Ramesh and Kenneth Li. 2025. Communicating activations between language model agents. In Forty-second International Conference on Machine Learning, ICML, volume 267 of Proceedings ofMachine Learning Research, Vancouver, BC.

Anna Rogers, Olga Kovaleva, Matthew Downey, and Anna Rumshisky. 2020. Getting closer to AI complete question answering: A set of prerequisite real tasks. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI, pages 8722–8731.

Salesforce AI Research. 2024. SFR-Embedding-Mistral: Model card.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social iqa: Commonsense reasoning about social interactions. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP, pages 4462–4472, Hong Kong, China.

Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., and Dejan Kostic. 2026. Kvcomm: Enabling efficient LLM communication through selective KV sharing. In The Fourteenth International Conference on Learning Representations, ICLR, Rio de Janeiro, Brazil.

Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, and Claire Cardie. 2019. Dream: A challenge data set and models for dialogue-based reading comprehension. Transactions ofthe Associationfor Computational Linguistics, 7:217–231.

Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. 2025. Mixture-of-agents enhances large language model capabilities. In The Thirteenth International Conference on Learning Representations, ICLR, Singapore.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. 2017. Crowdsourcing multiple choice science questions. In Proceedings ofthe 3rd Workshop on Noisy User-generated Text, NUT@EMNLP, pages 94–106, Copenhagen, Denmark.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. 2023. Autogen: Enabling next-gen LLM applications via multi-agent conversation framework.

Qizhe Xie, Guokun Lai, Zihang Dai, and Eduard H. Hovy. 2018. Large-scale cloze test dataset created by teachers. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 2344–2356.

Fu-Chun Yang and Jason Eshraghian. 2025. Direct semantic communication between large language models via vector translation. CoRR, abs/2511.03945.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR, Kigali, Rwanda.

Howard Yen, Tianyu Gao, and Danqi Chen. 2024. Longcontext language modeling with parallel context encoding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, , August 11-16, 2024, pages 2588–2610, Bangkok, Thailand.

Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. 2020. Reclor: A reading comprehension dataset requiring logical reasoning. In 8th International Conference on Learning Representations, ICLR, Addis Ababa, Ethiopia.

Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. 2018. SWAG: A large-scale adversarial dataset for grounded commonsense inference. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 93–104.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL, pages 4791–4800, Florence, Italy.

Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, and Rui Wang. 2026. F2llm-v2: Inclusive, performant, and efficient embeddings for a multilingual world. CoRR, abs/2603.19223.

Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, and Ling Yang. 2025. Latent collaboration in multiagent systems. CoRR, abs/2511.20639.

## A Implementation Details

Layer pairing. Equations use row vectors and omit the batch dimension. Let $n _ { E } , n _ { T }$ be the context encoder and receiver token counts, and $d _ { E } , d _ { T }$ their hidden dimensions. We choose $P$ unique layer pairs $( a _ { i } , b _ { i } )$ , with blocks selected uniformly across each model’s depth and paired in order. When $P > 1$ , the schedule includes the final blocks. Each pair has its own compressor and reader, and may have a different slot count $S _ { i }$ . In the six model pairs of Table 4, every receiver layer has a reader, with one exception: the Qwen3.5-0.8B encoder has 24 layers and the Gemma-3-1B-pt receiver has 26, so 24 receiver layers have a reader. The trainable parameters are $\theta = ( \phi , \psi , \eta )$ for context compressors, readers, and gates, respectively. Both backbone parameter sets are fixed.

Context compressor. Equation 1 uses $W _ { i } ^ { E } \ \in$ $\mathbb { R } ^ { d _ { E } \times d }$ and masks encoder padding positions in cross-attention. The learned-query residual is retained. The feed-forward network expands to $2 d ,$ applies SiLU, and projects back to $d \colon$

F $\mathrm { { } ^ { T F N } } _ { i } ( B ) = \mathrm { { S i L U } } ( B W _ { i , 1 } + b _ { i , 1 } ) W _ { i , 2 } + b _ { i , 2 } .$ (6) Here $W _ { i , 1 } \in \mathbb { R } ^ { d \times 2 d } , W _ { i , 2 } \in \mathbb { R } ^ { 2 d \times d } , b _ { i , 1 } \in \mathbb { R } ^ { 1 \times 2 d } ,$ and $b _ { i , 2 } \in \mathbb { R } ^ { 1 \times d }$ . Compressors and readers use the same attention-head count M, which must divide d. The implementation sets $M = 8$

Pooling and gates. Let $\mathcal { T } ( r )$ be the unmasked prompt positions, and $h _ { i , t }$ the receiver block output before the current reader. A shared attention pool summarizes these states:

$$
\begin{array} { r } { \bar { h } _ { i , t } = \mathrm { L N } _ { g } ( h _ { i , t } ) , } \end{array}\tag{7}
$$

$$
e _ { i , t } = \mathrm { t a n h } ( \bar { h } _ { i , t } W _ { p } + b _ { p } ) w _ { p } ,\tag{8}
$$

$$
\alpha _ { i , t } = \frac { \exp ( e _ { i , t } ) } { \sum _ { v \in \mathbb { Z } ( r ) } \exp ( e _ { i , v } ) } ,\tag{9}
$$

$$
p _ { i } ( x , r ) = \sum _ { t \in \mathcal { T } ( r ) } \alpha _ { i , t } \bar { h } _ { i , t } .\tag{10}
$$

Here $W _ { p } \in \mathbb R ^ { d _ { T } \times d _ { p } } , b _ { p } \in \mathbb R ^ { 1 \times d _ { p } }$ , and $w _ { p } \in \mathbb { R } ^ { d _ { p } \times 1 }$ where $d _ { p }$ is the hidden width of the pooling network. LayerNorm and pooling weights are trainable and shared across all pairs. Response and padding positions have zero pooling weight. The summary pools normalized contextual states rather than input embeddings. The gate projections are

$$
u _ { i } ( x , r ) = p _ { i } ( x , r ) W _ { g } ^ { Q } ,\tag{11}
$$

$$
k _ { i , s } ( x ) = z _ { i , s } ( x ) W _ { g } ^ { K } .\tag{12}
$$

Here $p _ { i } ( x , r ) \in \mathbb { R } ^ { 1 \times d _ { T } } , z _ { i , s } ( x ) = Z _ { i } ( x ) [ s , : ] \ \in$ $\mathbb { R } ^ { 1 \times d } , W _ { g } ^ { Q } \in \mathbb { R } ^ { d _ { T } \times d }$ , and $W _ { g } ^ { K } \in \mathbb { R } ^ { d \times d }$ . These bias-free projections are shared across all pairs and slots. These projections give the prompt– slot comparison in Equation 2. Gate keys $k _ { i , s }$ use a separate projection from reader attention keys $K _ { i }$ . Independent biases $\beta _ { i , i }$ <sub>s</sub> are initialized to 2. Including pooling, the gate module has $\begin{array} { r } { d _ { T } d + d ^ { 2 } + \sum _ { i } S _ { i } + d _ { T } d _ { p } + 2 d _ { p } + 2 d _ { T } } \end{array}$ parameters. The implementation sets $d _ { p } = 2 5 6$ and computes pooling and gate logits in bfloat16, without an additional LLM forward pass. Inference caches gates separately for each paired layer during prompt prefill; training retains the contextual gradient path, including earlier readers.

Reader. In Equation 3, $W _ { i } ^ { Q } ~ \in ~ \mathbb { R } ^ { d _ { T } \times d }$ and $W _ { i } ^ { K } , W _ { i } ^ { V } \in \mathbb { R } ^ { d \times d }$ . All three projections are biasfree; receiver and latent LayerNorms are separate. For head m, $Q _ { i } ^ { ( m ) } \in \mathbb { R } ^ { n _ { T } \times d _ { h } }$ and $K _ { i } ^ { ( m ) } , \bar { V _ { i } ^ { ( m ) } } \in$ $\mathbb { R } ^ { S _ { i } \times d _ { h } }$ . Attention has shape $A _ { i } ^ { ( m ) } \in \mathbb { R } ^ { n _ { T } \times S _ { i } }$ , and $G _ { i } \in \mathbb { R } ^ { S _ { i } \times S _ { i } }$ . Thus the gated read in Equation 3 is well-defined and produces $R _ { i } ^ { ( m ) } \in \mathbb { R } ^ { n _ { T } \times d _ { h } }$ . Writing $a _ { i , t , s } ^ { ( m ) } = A _ { i } ^ { ( m ) } [ t , s ]$ and $v _ { i , s } ^ { ( m ) } = V _ { i } ^ { ( m ) } [ s , : ]$ , the

output for token t is

$$
r _ { i , t } ^ { ( m ) } = \sum _ { s = 1 } ^ { S _ { i } } a _ { i , t , s } ^ { ( m ) } g _ { i , s } v _ { i , s } ^ { ( m ) } .\tag{13}
$$

Attention coefficients sum to one across slots; their gated coefficients are not renormalized. The same $G _ { i }$ is used for all tokens and heads. It is notation for row-wise scaling, implemented by broadcasting rather than constructing a dense matrix. The output projection $W _ { i } ^ { O } \in \bar { \mathbb { R } } ^ { d \times d _ { T } }$ is bias-free and initialized to zero, so the initial residual update is zero. Attention dropout is zero.

Supervision and payload. Reconstruction targets include the end-of-sequence token. Equation 4 averages over supervised response tokens, rather than giving equal weight to responses of different lengths. The context encoder LLM is evaluated without gradients. In Stage 1, gradients through the frozen receiver update the compressors, readers, and gates. In Stage 2, they update only the readers and gates. If a stored scalar occupies w bytes, the packet’s payload is $w d \textstyle \sum _ { i } S _ { i }$ bytes, excluding metadata. Gate and reader parameters are stored at the receiver rather than transmitted in the packet.

Training protocol implementation. Stage 2 loads the complete Stage 1 interface, including the prompt-pooling and gate parameters, from the final Stage 1 checkpoint with strict key matching, and starts a new optimizer and learning-rate schedule. The model pair, layer pairing, slot count, and latent width are identical in both stages, so every interface tensor transfers without reshaping or reinitialization. The compressor parameters are then frozen. Packets for the training contexts are computed once with the frozen encoder and compressors; identical contexts share one stored packet. The packets are stored in bfloat16. Stage 2 training reads the packets and does not load the context encoder LLM.

Training settings. Table 9 lists the training hyperparameters. Stage 1 runs one epoch over the reconstruction passages. Stage 2 then fixes the compressors, stores one packet for each training context, and runs three epochs over the task training set. The readers and gates that it trains have 16.4M parameters for the Qwen3.5-0.8B receiver. Both stages use AdamW with the default momentum coefficients. In Stage 1, the learning rate rises linearly for 157 updates and then stays constant. In Stage 2, it rises linearly for 45 updates and then follows a half-cosine decay to zero. The trained baselines use the same task data, loss, batch size, number of epochs, and learning-rate schedule. Every run uses random seed 42, and Stage 2 of the default pair is repeated with seeds 43 and 44.

<table><tr><td>Dataset</td><td>Train examples</td><td>Train context tokens</td><td>Avg. context</td><td>Eval</td></tr><tr><td colspan="5">Stage 1: context reconstruction</td></tr><tr><td>WikiText-103</td><td>446,282</td><td>100,278,346</td><td>224.7</td><td>1,057</td></tr><tr><td>FineWeb-Edu</td><td>1,553,718</td><td>1,033,473,862</td><td>665.2</td><td>13,926</td></tr><tr><td>Stage 1 total</td><td>2,000,000</td><td>1,133,752,208</td><td>566.9</td><td>14,983</td></tr><tr><td colspan="5">Stage 2: task supervision</td></tr><tr><td>CLOTH</td><td>76,660</td><td>34,320,807</td><td>447.7</td><td>11,516</td></tr><tr><td>Cosmos QA</td><td>25,067</td><td>1,861,200</td><td>74.2</td><td>2,985</td></tr><tr><td>HellaSwag</td><td>31,308</td><td>1,472,101</td><td>47.0</td><td>10,042</td></tr><tr><td>LogiQA</td><td>7,202</td><td>578,041</td><td>80.3</td><td>651</td></tr><tr><td>MCScript2.0</td><td>14,017</td><td>2,372,120</td><td>169.2</td><td>3,610</td></tr><tr><td>MuTual</td><td>6,901</td><td>685,106</td><td>99.3</td><td>886</td></tr><tr><td>OpenBookQA</td><td>3,205</td><td>33,160</td><td>10.3</td><td>500</td></tr><tr><td>QuALITY</td><td>2,393</td><td>14,215,251</td><td>5,940.3</td><td>2,086</td></tr><tr><td>ReClor</td><td>4,422</td><td>351,546</td><td>79.5</td><td>500</td></tr><tr><td>SciQ</td><td>10,210</td><td>992,369</td><td>97.2</td><td>884</td></tr><tr><td>SocialIQA</td><td>33,232</td><td>554,725</td><td>16.7</td><td>2,224</td></tr><tr><td>SWAG</td><td>67,824</td><td>1,118,265</td><td>16.5</td><td>20,006</td></tr><tr><td>Stage 2 total</td><td>282,441</td><td>58,554,691</td><td>207.3</td><td>55,890</td></tr><tr><td colspan="5">Held-out: test only</td></tr><tr><td>RACE</td><td></td><td></td><td>343.6</td><td>4,934</td></tr><tr><td>DREAM</td><td>一</td><td></td><td>156.4</td><td>2,041</td></tr><tr><td>QuAIL</td><td>一</td><td></td><td>419.6</td><td>2,720</td></tr></table>

Table 8: Data for both training stages and the heldout datasets. Train excludes the development holdout. Tokens count untruncated context text once per training example with the pinned Qwen3.5-0.8B tokenizer, excluding instructions, choices, targets, and special tokens. Reconstruction supervision contains the same text plus one EOS per passage (1,135,752,208 target tokens). Avg. context is the mean training context length in tokens, rounded to one decimal; total-row means are weighted by example count. For the held-out datasets, it is the mean context length of the test questions. Eval is the reconstruction test set, the fixed public multiplechoice reporting suite, or the held-out test set.

Environment. LatComm runs use Python 3.12, Py-Torch 2.11 or 2.13, and Hugging Face Transformers 5.15.1. The frozen models and the interface modules run in 16-bit bfloat16, and packets are stored in the same format. All experiments run on six machines, each with eight NVIDIA H100 GPUs (80 GB each). The trained interfaces are evaluated with Transformers. The Raw-text reference on the 12 task datasets is computed with vLLM.

## B Data Details

Reconstruction data. The Stage 1 training set contains 2,000,000 passages: 446,282 from WikiText-

<table><tr><td>Setting</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Data</td><td></td><td></td></tr><tr><td>Training examples</td><td>2,000,000</td><td>282,441</td></tr><tr><td>Epochs</td><td>1</td><td>3</td></tr><tr><td>Updates</td><td>5,209</td><td>2,208</td></tr><tr><td>Batch size</td><td>384</td><td>384</td></tr><tr><td>Optimization</td><td></td><td></td></tr><tr><td>Trained modules</td><td>all</td><td>readers, gates</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td> $1 0 ^ { - 4 }$ </td><td>10⁻4</td></tr><tr><td>Warmup updates</td><td>157</td><td>45</td></tr><tr><td>Schedule after warmup</td><td>constant</td><td>half-cosine to 0</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td></tr><tr><td>Gradient-norm clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Random seed</td><td>42</td><td>42</td></tr><tr><td>Interface</td><td></td><td></td></tr><tr><td>Slots per layer pair S</td><td>32</td><td></td></tr><tr><td>Slot width d</td><td></td><td></td></tr><tr><td>Attention heads M</td><td></td><td></td></tr><tr><td>Pooling width  $d _ { p }$ </td><td></td><td></td></tr><tr><td>Initial gate bias  $\beta _ { i , s }$ </td><td></td><td></td></tr><tr><td>Precision</td><td></td><td></td></tr><tr><td>Frozen LLMs</td><td>bfloat16</td><td></td></tr><tr><td>Interface modules</td><td></td><td>bfloat16</td></tr></table>

Table 9: Training hyperparameters of the default configuration. Stage 1 trains the compressors, readers, and gates, and Stage 2 trains only the readers and gates.

103 and 1,553,718 from FineWeb-Edu. They are a fixed selection of complete passages from the prepared corpus. Each passage contains 64–1,024 tokens under the Qwen3.5-0.8B tokenizer. The receiver is asked to reproduce the passage with one of 100 reconstruction instructions. Validation and test contain 15,043 and 14,983 passages. WikiText-103 keeps its official splits, and FineWeb-Edu is split by canonical URL before passage extraction. English filtering, passage deduplication, and document-level separation are applied.

Task data. Table 8 lists the 12 task datasets. After removing exact normalized context overlaps with the test set and invalid duplicate answer texts, the official training pool contains 284,477 questions. A grouped holdout from this pool gives 2,036 development questions, and the remaining 282,441 questions are used for training. The test set contains 55,890 labeled questions from public benchmark splits. These questions are not used for training or model selection. The original option text, option count, and option order are kept. SciQ uses a fixed deterministic option order because its source does not specify one. MuTual uses its labeled development split for testing because its official test labels are not public. We also compute Micro, the accuracy over all 55,890 test questions.

![](images/d280798628a693ac7d5bdd6d1af0b2508841e9945013ea02cafc5e859279583c.jpg)

(a) Stage 1: context tokens processed  
![](images/81746ada1b4da5c0b6528659bd2a05f4c1f319b690b19f733faf42fb7dfea04b.jpg)  
(b) Stage 2: context tokens processed

Figure 5: Training loss of (a) Stage 1 and (b) Stage 2 against the number of context tokens processed, for the configurations of Table 10. Each column changes one setting, and the solid orange line is the default configuration. The token count is the number of processed examples times the mean context length of the training set. The curves are running means over 50 updates in Stage 1 and 100 updates in Stage 2. Dotted lines mark the ends of the first and second epochs of Stage 2.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Packet</td><td rowspan="2">Size</td><td colspan="2">Reconstruction</td><td rowspan="2">Task Avg</td></tr><tr><td>Loss</td><td>Acc.</td></tr><tr><td>Encoder</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-0.8B</td><td> $2 4 \times 3 2 \times 2 5 6$ </td><td>384 KiB 1.83</td><td></td><td>56.3</td><td>64.91</td></tr><tr><td>Qwen3.5-4B (default)</td><td> $2 4 \times 3 2 \times 2 5 6$ </td><td>384 KiB</td><td>1.24</td><td>68.5</td><td>65.43</td></tr><tr><td>Qwen3.5-9B</td><td> $2 4 \times 3 2 \times 2 5 6$ </td><td>384 KiB</td><td>0.95</td><td>75.1</td><td>65.80</td></tr><tr><td>Slots per pair</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td> $2 4 \times 8 \times 2 5 6$ </td><td>96KiB</td><td>1.58</td><td>61.3</td><td>65.81</td></tr><tr><td>32 (default)</td><td> $2 4 \times 3 2 \times 2 5 6$ </td><td>384 KiB</td><td>1.24</td><td>68.5</td><td>65.43</td></tr><tr><td>64</td><td> $2 4 \times 6 4 \times 2 5 6$ </td><td>768 KiB</td><td>1.11</td><td>71.5</td><td>65.04</td></tr><tr><td>Slot width</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>128</td><td> $2 4 \times 3 2 \times 1 2 8$ </td><td>192 KiB</td><td>1.82</td><td>57.1</td><td>65.01</td></tr><tr><td>256 (default)</td><td> $2 4 \times 3 2 \times 2 5 6$ </td><td>384 KiB</td><td>1.24</td><td>68.5</td><td>65.43</td></tr><tr><td>512</td><td> $2 4 \times 3 2 \times 5 1 2$ </td><td>768 KiB</td><td>0.78</td><td>79.0</td><td>66.88</td></tr><tr><td>Layer pairs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td> $6 \times 3 2 \times 2 5 6$ </td><td>96KiB</td><td>1.77</td><td>57.9</td><td>63.84</td></tr><tr><td>12</td><td> $1 2 \times 3 2 \times 2 5 6$ </td><td>192 KiB</td><td>1.47</td><td>63.5</td><td>64.76</td></tr><tr><td>24 (default)</td><td> $2 4 \times 3 2 \times 2 5 6$ </td><td>384 KiB</td><td>1.24</td><td>68.5</td><td>65.43</td></tr></table>

Table 10: Ablation of the context encoder and the packet settings with the Qwen3.5-0.8B receiver. Each group changes one setting of the default configuration. Packet gives $P \times S \times d \colon$ layer pairs, slots per pair, and slot width. Size is the stored size. Reconstruction gives the loss and the token accuracy (%) on 256 held-out passages at the end of Stage 1. Task Avg (%) is measured after Stage 2.

![](images/5111033f809c29cb732559b6d5e04d554f2e2f9659548b992be9733d2ff041ea.jpg)  
Figure 6: Accuracy on the test questions that the Qwen3.5-4B encoder answers correctly or wrongly from the full text. Embedding uses BGE-Gemma2.

Held-out datasets. RACE, DREAM, and QuAIL are used only for testing. RACE and DREAM use their official test sets. QuAIL uses its validation and challenge sets. No held-out passage is an exact match of a Stage 2 training context. Some RACE passages share text with CLOTH training passages, and some DREAM dialogues share text with MuTual training dialogues, because these datasets draw on similar exam material. Removing the affected questions (294 in RACE and 349 in DREAM) changes the accuracy of every method by less than one point.

<table><tr><td>Packet given to the receiver</td><td colspan="10">Task datasets (%)</td><td colspan="5">Held-out datasets (%)</td></tr><tr><td></td><td></td><td>CLOTH Cosmos HSwag LogiQA MCScr. MuTual OBQA QuAL. ReClor SciQ SIQA SWAG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg</td><td></td><td>RACE DREAM QuAIL</td><td></td><td>Avg</td></tr><tr><td colspan="10">Same trained readers and gates</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Own packet</td><td>52.2</td><td>61.8</td><td>73.4</td><td>37.9 80.8</td><td>73.9</td><td>79.6</td><td>43.2</td><td>49.8</td><td>92.3 66.5</td><td>73.2</td><td>65.39</td><td>52.3</td><td>52.5</td><td>44.6</td><td>49.82</td></tr><tr><td>Packet of another context</td><td>30.7</td><td>48.1</td><td>50.3 28.1</td><td>63.4</td><td>43.8</td><td>51.2</td><td>31.7</td><td>30.6 77.3</td><td>45.4</td><td>46.4</td><td>45.59</td><td>37.8</td><td>38.5</td><td>32.5</td><td>36.28</td></tr><tr><td>No packet</td><td>25.0</td><td>44.8</td><td>28.9</td><td>26.0 62.6</td><td>32.3</td><td>53.6</td><td>32.2</td><td>27.8</td><td>77.1 37.7</td><td>27.3</td><td>39.60</td><td>39.1</td><td>36.2</td><td>34.0</td><td>36.41</td></tr><tr><td colspan="10">Readers and gates trained with one fixed packet</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Constant packet</td><td>44.2</td><td>63.2</td><td>68.9</td><td>37.5</td><td>75.6 71.9</td><td>68.8</td><td>39.5</td><td></td><td></td><td>41.8 88.1 57.6 65.8 60.25</td><td></td><td>45.5</td><td>41.2</td><td></td><td>43.543.37</td></tr></table>

Table 11: Packet controls for the default configuration: accuracy on each dataset. The first three rows use the same trained readers and gates and change only the packet. The packet of another context comes from a different test question. The three rows come from separate evaluation runs, so the first row differs slightly from Table 3. Columns and abbreviations follow Table 3.

![](images/50e8847251ba67ac9f9a8820da09a64522fa061f50027e7113c7dbb91f3de05d.jpg)  
Figure 7: Receiver time per question against the context length on QuALITY. Each point is one article.

Token counts. Table 8 counts context tokens with the Qwen3.5-0.8B tokenizer, before truncation and without instructions, answer choices, responses, or special tokens. A context shared by several questions is counted once for each question. The reconstruction training set contains 1,133,752,208 context tokens, and the task training set contains 58,554,691. The cross-family pairs use the same texts with their own tokenizers.

## C Baseline Details

Constant packet. This control uses the receiver, readers, gates, Stage 1 initialization, data, and training settings of the corresponding LatComm pair: Qwen3.5-4B for the Qwen3.5-0.8B receiver, Qwen3.5-9B for the Gemma-3-1B-pt receiver, and Granite-4.1-8B for the Llama-3.2-1B receiver. The only change is the packet. We encode 2,048 randomly chosen training contexts with the pair’s Stage 1 compressors and average their packets. Every training and test question receives this mean packet, so the packet carries no information about the question’s context.

Embedding. This baseline compresses the context into one vector with a frozen text embedding model. We use F2LLM-v2-14B (Zhang et al., 2026) and BGE-Multilingual-Gemma2 (9B) (Beijing Academy of Artificial Intelligence, 2024), each paired with the frozen Qwen3.5-0.8B and Gemma-3-1B-pt receivers. The encoder processes the context as one input, without the question and without chunk pooling. Its last-token representation is L2-normalized, which gives a 5,120-dimensional vector for F2LLM and a 3,584-dimensional vector for BGE. A trainable shared projector maps this vector to one slot. The readers and gates are the same as in LatComm, and they read this slot at every receiver layer: 24 layers for Qwen3.5-0.8B and 26 for Gemma-3-1B-pt. The stored 32-bit vectors take 20 KiB and 14 KiB. Each interface is trained from a fresh initialization on the task data, without a reconstruction stage. F2LLM uses a 32,768-token input limit, which truncates no test context. BGE truncates its input to 8,192 tokens. This affects 93 of the 55,890 test questions, which share five documents. For Table 6, BGE is paired with the Llama-3.2-1B receiver (16 layers). In this run, the BGE vectors are computed in bfloat16 and stored, instead of in 32-bit precision during training. On 15 documents of different lengths, the two computations give a cosine similarity above 0.999.

xRAG. We adapt xRAG (Cheng et al., 2024). A frozen SFR-Embedding-Mistral encoder (Salesforce AI Research, 2024) reads the complete context, and a two-layer MLP maps its 4,096- dimensional last-token representation to one soft token placed before the receiver’s prompt. Only the MLP is trained, first on reconstruction and then on the task data. Our adaptation differs from the original in three ways: there is no retrieval or document selection, the receiver is Qwen3.5-0.8B, and task training uses the gold option letter without a distillation term. The second xRAG variant replaces the embedding model with the last-token state of Qwen3.5-4B, which has 2,560 dimensions.

![](images/f3645abd24bcff6934e3e3d79917792ad318570d2b4b5c916a13a6033e9cbd99.jpg)  
(a) Gate value per layer pair

![](images/da520f1aa1522d3e3336e3b85fc671aed913837b134f35351f0f3ed3688d5987.jpg)  
(b) Mean gate of each slot

Figure 8: Gate values of the default pair on 5,386 test questions. (a) Mean over the 32 slots of each layer pair. The band covers the 10th to 90th percentile of the questions. (b) Mean gate of each slot. The slots of a layer pair are sorted from the highest mean gate to the lowest.  
![](images/78982874c4b137ebc7e8d8c6687cd633a3054e09087d06603d8d429473f59332.jpg)  
(a) Between slots of one packet

![](images/f758917c5a048010a83c425dee4d58f8cc9bfc77289be996362fb0717a699cad.jpg)  
(b) Between packets of two contexts

![](images/cd51257e8aea1d12c0fb66d670316d1e4f3cbbee49c277f9c20595eb2c3b7322.jpg)  
(c) Dataset identified from the packet  
Figure 9: Packets of the default pair at each layer pair. (a) Mean cosine similarity between two slots of one packet. (b) Mean cosine similarity between the packets of two contexts from the same dataset or from different datasets. (c) Balanced accuracy of identifying the source dataset of a context from its packet.

## D Ablation Study

Packet settings. Table 10 changes the context encoder or one packet setting at a time. The packet settings use the Qwen3.5-4B encoder and change the number of slots (8, 32, 64), the slot width (128, 256, 512), and the number of layer pairs (24, 12, 6). Reconstruction improves with every increase in capacity: a larger encoder, more slots, wider slots, and more layer pairs all lower the Stage 1 loss and raise the token accuracy. The widest slots give the best reconstruction, with 79.0% token accuracy, and the highest Task Avg, 66.88. Task Avg changes less than reconstruction. It stays between 63.84 and 66.88 over all configurations, and every configuration is above the best single-vector baseline (61.98). With 8 slots per pair, the packet takes

96 KiB, one quarter of the default size, and Task Avg is 65.81, against 65.43 for the default packet. With 12 and 6 layer pairs, Task Avg is 64.76 and 63.84. At 96 KiB, a packet with all layer pairs and 8 slots is therefore more accurate than a packet with 6 layer pairs and 32 slots.

Training loss. Figure 5 shows the training loss of both stages for the same configurations. In Stage 1, the loss decreases throughout training, and the configurations keep the order of Table 10: more capacity gives a lower reconstruction loss. In Stage 2, the curves are close to each other for the three encoders and for the three slot counts, and the loss is lower with wider slots and with more layer pairs.

Packet controls. Table 11 tests whether the receiver uses the content of the packet. The same trained readers and gates reach 65.39 Task Avg with the packet of the question’s own context, 45.59 with the packet of another context, and 39.60 without a packet. The own packet is the most accurate of the three on all 12 task datasets, and the packet of another context lowers the accuracy by 9.8 to 30.1 points. On the held-out datasets, the own packet reaches 49.82 Held-out Avg, against 36.28 with the packet of another context and 36.41 without a packet. A packet from another context therefore gives no gain on new datasets, and the gain comes from the content of the own packet. The Constant packet control, whose readers and gates are trained with one fixed packet, reaches 60.25 Task Avg and 43.37 Held-out Avg.

## E Additional Analysis

Seed variation. We repeat Stage 2 of the default pair with three seeds. The three runs share the compressors and the packets. Task Avg is 65.43, 65.72, and 65.88, with a mean of 65.68 and a standard deviation of 0.23. The main tables report the first run, which is the lowest of the three, and the gain over the best baseline (61.98) is much larger than the variation between seeds.

Gain source. We split the test questions of the 12 task datasets by whether the Qwen3.5-4B encoder answers them correctly when it reads the full text, which holds for 76% of the questions (Figure 6). On these questions, LatComm reaches 72.4% accuracy, against 68.7% for the Embedding baseline and 64.5% for the Constant packet. On the other questions, the three methods are close, with 50.5%, 48.0%, and 50.1%. The gain of LatComm is therefore on the questions that the encoder itself can answer from the context.

Context length. Figure 7 plots the receiver time per question for each of the 115 QuALITY articles. With text passing, the time grows linearly with the context length, from 265 ms for the shortest article to 707 ms for the longest. With a packet, it stays between 130 and 135 ms for every article, because the receiver always reads the same number of slots. Gate values. We record the gates of the default pair on a sample of 5,386 test questions (Figure 8). The mean gate is 0.40, and it differs across layer pairs, from 0.05 at the last pair to 0.59 at pair 18. The slots of one layer pair also differ: 23.4% of the layer–slot cells have a mean gate below 0.05, and 4.3% have a mean gate above 0.95. The receiver therefore weights the slots differently at each layer pair, which one gate per layer could not express.

Packet similarity. Figure 9 compares the packets of the contexts in the same sample. The slots of one layer pair are nearly orthogonal at the first layer pairs, with a mean cosine similarity of 0.04, and more similar at the last ones, with 0.65. The packets of two contexts from different datasets have a mean cosine similarity of 0.72, so a part of the packet is shared by all contexts, which agrees with the accuracy of the Constant packet control. The other part separates the contexts. Packets from the same dataset are more similar than packets from different datasets at every layer pair. A nearestmean classifier on the mean slot of a layer pair, fitted on half of the contexts and tested on the rest, identifies the source dataset among 12 with 30% to 92% balanced accuracy, against 8.3% by chance.