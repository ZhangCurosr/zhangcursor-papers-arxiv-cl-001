# Variational Quantum Transformer Architecture for Synthetic Language Generation

Julian Hager<sup>1[0000−0001−8220−4522]</sup>, Michael Kölle<sup>1[0000−0002−8472−9944]</sup>, Gerhard Stenzel<sup>1[0009−0009−0280−4911]</sup>, Tobias Rohe<sup>1[0009−0003−3283−0586]</sup>, Jonas Stein<sup>1[0000−0001−5727−9151]</sup>, and Claudia Linnhof-Popien<sup>1[0000−0001−6284−9286]</sup>

Institute of Informatics, LMU Munich, Oettingenstraße 67, 80538 Munich, Germany julian.hager@ifi.lmu.de

Abstract. We propose a compact NISQ-compatible quantum transformer architecture for synthetic QNLP sequence modelling. The model preserves the autoregressive next-token interface of a classical transformer, but replaces attention and feed-forward sublayers with variational quantum encoder blocks, connector circuits, decoder blocks and a direct two-qubit measurement readout. Token contexts are angle-encoded into small quantum registers, processed by parallel variational heads and encoder integration circuits and conditioned through decoder ancillae to produce a distribution over a four-token vocabulary. We evaluate several architecture variants on deterministic and lexicographic grammargeneration tasks against a compact classical transformer baseline. The quantum models are trainable end-to-end and learn nontrivial grammar structure, including perfect deterministic generation in individual runs and high lexicographic validity in the strongest variant. The classical baseline remains more accurate and stable and the quantum models are sensitive to initialization. The contribution is therefore not a claim of quantum advantage, but a concrete architecture and evaluation of transformer-inspired QNLP sequence modelling under near-term quantum constraints.

Keywords: Quantum Natural Language Processing · Quantum Transformer · Variational Quantum Circuits · NISQ · Sequence Modelling

## 1 Introduction

Transformers are the dominant architecture for sequence modelling because they expose a practical autoregressive interface while integrating contextual information through trainable internal blocks [1]. Quantum natural language processing (QNLP) ofers a complementary perspective in which linguistic structure is represented through tensorial or circuit-like models [6,7]. This raises a natural architectural question: can a transformer-style next-token model be expressed using compact variational quantum circuits while remaining compatible with near-term quantum constraints?

This question is nontrivial in the NISQ setting, where useful models must use few qubits, moderate circuit depth and measurement interfaces that can be trained in a hybrid loop [2,3,5]. We address it by proposing a NISQ-compatible quantum transformer for synthetic QNLP sequence modelling. The model maps a fixed context window to a next-token distribution, but replaces classical attention and feed-forward sublayers with variational quantum encoder blocks, connector circuits, decoder blocks and a direct density-matrix readout. The term transformer is therefore used architecturally. The model retains heads, encoder integration, decoder conditioning and autoregressive prediction, but implements these stages with quantum circuits.

Related work has also explored quantum natural language generation on nearterm devices [12]. Here we study a complementary architecture-driven setting based on autoregressive variational quantum circuits. We evaluate the architecture on two controlled grammar tasks over a four-token vocabulary: a deterministic cycle, AAA BBB CCC, and a lexicographic language of nondecreasing length-three words. These tasks are deliberately small. Their purpose is to isolate whether the architecture can learn explicit sequence structure under conditions where grammatical validity is measurable. Compared with a compact classical transformer baseline, the quantum variants learn nontrivial grammar regularities and can solve the deterministic task in individual runs, but remain less accurate and less stable overall. Thus this work does not claim quantum advantage. Its contributions are: (i) a concrete variational encoder–connector–decoder architecture for autoregressive QNLP experiments, (ii) parameter-eficient NISQ-scale variants with small active circuit width and (iii) a controlled evaluation using token-level and grammar-level metrics.

## 2 Background and Motivation

Transformers and QNLP. A transformer maps token representations to contextdependent features and, in autoregressive use, to next-token probabilities [1]. We use this interface as a design pattern rather than reproducing scaled dot-product attention directly. Parallel heads produce intermediate representations, an encoder integrates them, a decoder conditions generation on the encoder state and the final state is measured as a token distribution. This connects to QNLP, where linguistic composition has been studied through tensor-network or circuit-like representations [6,7], practical near-term executions [13] and software pipelines for mapping language to circuits [14]. Recent work has also explored quantum attention and transformer-like models [9,10,11].

Variational circuits under NISQ constraints. Variational quantum circuits encode data into quantum states, transform them with trainable gates and optimize parameters from measurement-derived outputs [3,5]. They are a natural model class for NISQ studies, but they impose architectural pressure by requiring small circuit width and depth together with a readout simple enough for repeated optimization [2]. These constraints motivate the choices made below, including small active registers, shallow strongly entangling layers, repeated reduction to two-qubit states and a direct measurement-based output rather than a large learned classical head. The aim is not to replace classical transformers at scale, but to define a compact architecture whose sequence-modelling behaviour can be studied under near-term assumptions.

## 3 A NISQ-Compatible Quantum Transformer

The proposed model is an autoregressive hybrid quantum sequence model. Given a context window $x _ { t - c : t - 1 }$ , it estimates

$$
p _ { \Theta } ( x _ { t } = k \ | \ x _ { t - c : t - 1 } ) , \qquad k \in \mathcal { V } ,\tag{1}
$$

where Θ denotes the trainable circuit parameters. In the experiments, $\nu =$ $\{ A , B , C , \mathsf { s p a c e } \}$ and $c = 4$ . The output vocabulary therefore matches the four computational-basis probabilities of a two-qubit readout state. A detailed circuit diagram is provided in Appendix A. This section summarizes the architectural data flow.

## 3.1 Input Encoding

Following the view of quantum models as feature-space methods [4], tokens are mapped to integer identifiers $0 , \ldots , | \mathcal { V } | - 1$ and angle-encoded into a four-qubit register by applying one rotation per context position,

$$
| \psi _ { x } \rangle = \bigotimes _ { j = 1 } ^ { c } R _ { X } \left( { \frac { \pi } { 2 } } x _ { j } \right) | 0 \rangle , \qquad x _ { j } \in \{ 0 , 1 , 2 , 3 \} .\tag{2}
$$

The wire index represents the token position, so no separate positional encoding is used. For the four-token vocabulary and four-token context, this yields $4 ^ { 4 }$ distinguishable context encodings using four data qubits. We write the corresponding density matrix as $\rho _ { x }$

## 3.2 Encoder, Connector and Decoder

The encoder consists of E blocks. Each block contains two parallel variational head circuits followed by an integration circuit. In the first encoder layer, each head receives $\rho _ { x }$ on the four data qubits and appends two ancilla qubits. Later layers receive the two-qubit state produced by the previous encoder block. Each head applies V strongly entangling layers, implemented with parameterized rotations and CNOT entanglement in PennyLane [8], and returns a two-qubit reduced state. The integration circuit combines the two head states with the same variational template and again reduces the result to two qubits. After E blocks, the encoder state $\rho _ { E }$ is therefore a compact two-qubit representation of the complete context.

The decoder consists of D blocks and conditions generation on $\rho _ { E }$ . In each decoder layer, a connector circuit couples the encoder state to a two-qubit decoder ancilla state. The connector applies CNOT gates from each encoder-output qubit to each decoder-ancilla qubit, followed by one trainable $R _ { Y }$ rotation on each decoder ancilla. The resulting state is combined with the original context state $\rho _ { x }$ and processed by a variational decoder circuit over the two decoder ancilla qubits and four data qubits. The decoder output is again reduced to two qubits and passed to the next decoder layer. This design provides a residual-style path from the original context to every decoder layer while keeping the intermediate representation small.

## 3.3 Measurement Readout and Variants

After the final decoder layer, the model obtains a two-qubit density matrix $\sigma _ { D }$ Its diagonal entries are used directly as next-token probabilities,

$$
p _ { \Theta } ( x _ { t } = k \mid x _ { t - c : t - 1 } ) = \frac { [ \mathrm { d i a g } ( \sigma _ { D } ) ] _ { k } } { \sum _ { j = 0 } ^ { | \mathcal { N } | - 1 } [ \mathrm { d i a g } ( \sigma _ { D } ) ] _ { j } } , \qquad k \in \{ 0 , 1 , 2 , 3 \} .\tag{3}
$$

In exact arithmetic the denominator is one. The implementation clamps and renormalizes probabilities for numerical stability. Generation uses greedy autoregressive decoding by appending the most likely token to the context window.

All variants use two heads, four data qubits, two-qubit encoder and decoder outputs and a maximum active circuit width of six qubits. Variant names have the form $\mathsf { q } _ { - } \mathsf { e } E _ { - } \mathsf { d } D _ { - } \mathsf { v } V$ , indicating encoder depth, decoder depth and variational depth. For the fixed register sizes used here, the number of trainable parameters is

$$
P ( E , D , V ) = \big ( 4 8 + 3 6 ( E - 1 ) \big ) V + D ( 1 8 V + 2 ) ,\tag{4}
$$

which counts encoder heads, integration circuits, decoder circuits and connector rotations. The evaluated variants are $\mathsf { q _ { - } e 2 _ { - } d 2 _ { - } v 1 , q _ { - } e 2 _ { - } d 2 _ { - } v 2 , q _ { - } e 2 _ { - } d 2 _ { - } v 3 }$ and $\mathsf { q } _ { - } \mathsf { e } 3 _ { - } \mathsf { d } 3 _ { - } \mathsf { v } 2$ . Their parameter counts are reported in Table 1.

## 4 Evaluation on Synthetic QNLP Grammars

We evaluate the architecture on controlled grammar tasks rather than opendomain language. This keeps the setting small enough for repeated quantum simulation while making structural correctness explicit and measurable.

Tasks. All experiments use three letter tokens, A, B and $C ,$ plus a separator token, space. Texts are sequences of three-character words separated by space and tokenized at the character level. The deterministic grammar is the periodic word cycle

$$
G _ { \mathrm { d e t } } = ( \mathtt { A A A } , \mathtt { B B B } , \mathtt { C C C } ) ^ { * } ,\tag{5}
$$

which tests whether the model can learn both word-internal repetition and longer-range phase structure. The lexicographic grammar contains all lengththree words with nondecreasing characters,

$$
G _ { \mathrm { l e x } } = \{ a b c \in \Sigma ^ { 3 } \mid a \leq b \leq c \} .\tag{6}
$$

This task admits many valid continuations, so exact token agreement with a sampled test sequence is stricter than grammatical validity.

Training protocol and baseline. For each grammar, the training text contains 25 words and the test text contains 40 words. Generation starts from the first four test tokens and continues greedily for 100 tokens. We report means and standard deviations over seeds 17, 23 and 42. The same seed controls model initialization and training randomness. For the lexicographic task it also controls the sampled training text, with the held-out test text sampled from the following seed. Quantum variants are trained for 20 epochs with Adam and learning rate $1 0 ^ { - 2 }$ using negative log-likelihood loss. The classical baseline is a small encoder–decoder transformer with $d _ { \mathrm { m o d e l } } = 4$ , two encoder layers, two decoder layers, two attention heads, feed-forward dimension 2, dropout 0.1 resulting in 700 trainable parameters. It is trained for 800 epochs with Adam and learning rate $1 0 ^ { - 3 }$ . The longer schedule reflects the lower cost of classical batched training and is intended to provide a strong sanity-check baseline rather than a parameter-matched competitor.

Metrics. We report token accuracy on the generated continuation and a grammar score measuring structural validity. Token accuracy compares generated tokens with the held-out test sequence after the initial context. The grammar score averages four rule-based components like valid-token rate, separator-position rate, word-length validity and either deterministic-cycle validity for $G _ { \mathrm { d e t } }$ or lexicographic-word validity for $G _ { \mathrm { l e x } }$ . The score is a diagnostic complement to token accuracy. In particular, a lexicographic output such as repeated AAA can be formally valid while still having low diversity and low agreement with the sampled target sequence.

## 5 Results

Table 1 reports aggregate performance over the three seeds. Loss values are included for completeness but are most meaningful within a model family because the quantum and classical models use diferent training schedules. The token and grammar metrics are evaluated under the same autoregressive generation protocol.

Deterministic grammar. The classical transformer solves the deterministic task in all runs, reaching perfect token accuracy and grammar score. The quantum variants learn weaker but nontrivial structure. The smallest model, $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathrm { d } 2 _ { - } \mathsf { v } 1$ , obtains low token accuracy but a higher grammar score, indicating valid symbols and partial separator alignment without reliable recovery of the full cycle. Increasing variational depth improves performance. $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathrm { d } 2 _ { - } \mathsf { v } 3$ gives the strongest deterministic quantum result with grammar score $0 . 7 5 5 \pm 0 . 2 1 3 .$ The large standard deviations are significant. The $V \in \{ 2 , 3 \}$ variants solve the deterministic continuation perfectly in one seed but fail to do so consistently. Thus the architecture is expressive enough to represent the rule, but the current optimization procedure does not reliably find that solution.

Table 1. Performance of the quantum transformer variants and the classical transformer baseline on the synthetic grammar tasks. Values are mean ± standard deviation over three seeds.
<table><tr><td></td><td>Task Model</td><td>Params</td><td></td><td></td><td>Loss ↓ Token acc. ↑ Grammar score ↑</td></tr><tr><td>Det.</td><td> $\mathrm { C l a s s i c a l }$ </td><td>700</td><td> $0 . 0 4 9 { \pm } 0 . 0 2 0$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>Det.</td><td> $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathsf { d } 2 _ { - } \mathsf { v } 1$ </td><td>124</td><td> $1 . 0 4 1 { \pm } 0 . 0 3 3$ </td><td> $0 . 2 5 7 { \pm } 0 . 0 4 5$ </td><td> $0 . 5 6 1 { \pm } 0 . 0 2 9$ </td></tr><tr><td>Det.</td><td> $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathsf { d } 2 _ { - } \mathsf { v } 2$ </td><td>244</td><td> $0 . 8 5 7 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 5 6 7 { \scriptstyle \pm 0 . 3 7 9 }$ </td><td> $0 . 6 9 4 { \scriptstyle \pm 0 . 2 7 1 }$ </td></tr><tr><td>Det.</td><td> $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathsf { d } 2 _ { - } \mathsf { v } 3$ </td><td>364</td><td> $0 . 6 7 2 { \scriptstyle \pm 0 . 0 8 5 }$ </td><td> $0 . 5 5 7 { \pm } 0 . 3 8 4$ </td><td> $0 . 7 5 5 { \pm } 0 . 2 1 3$ </td></tr><tr><td>Det.</td><td> $\mathsf { q } _ { - } \mathsf { e } 3 _ { - } \mathsf { d } 3 _ { - } \mathsf { v } 2$ </td><td>354</td><td>0.926±0.081</td><td> $0 . 4 9 0 { \scriptstyle \pm 0 . 2 0 7 }$ </td><td> $0 . 6 4 6 { \scriptstyle \pm 0 . 0 9 6 }$ </td></tr><tr><td>Lex.</td><td> $\mathrm { C l a s s i c a l }$ </td><td>700</td><td> $0 . 5 8 3 { \pm } 0 . 0 3 1$ </td><td> $0 . 5 5 7 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>Lex.</td><td> $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathsf { d } 2 _ { - } \mathsf { v } 1$ </td><td>124</td><td> $1 . 2 9 8 { \pm } 0 . 0 5 2$ </td><td> $0 . 2 6 7 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 4 4 6 { \pm } 0 . 0 7 9$ </td></tr><tr><td>Lex.</td><td> $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathsf { d } 2 _ { - } \mathsf { v } 2$ </td><td>244</td><td> $1 . 1 5 6 { \pm } 0 . 0 4 8$ </td><td> $0 . 3 3 7 { \pm } 0 . 0 8 3$ </td><td> $0 . 5 3 2 { \pm } 0 . 1 3 6$ </td></tr><tr><td>Lex.</td><td> $\mathsf { q } _ { - } \mathsf { e } 2 _ { - } \mathsf { d } 2 _ { - } \mathsf { v } 3$ </td><td>364</td><td> $1 . 0 7 8 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 2 8 0 { \pm } 0 . 1 0 4$ </td><td> $0 . 6 5 4 { \pm } 0 . 1 8 8$ </td></tr><tr><td>Lex.</td><td> $\mathsf { q } _ { - } \mathsf { e } 3 _ { - } \mathsf { d } 3 _ { - } \mathsf { v } 2$ </td><td>354</td><td> $1 . 1 5 4 { \pm } 0 . 0 3 3$ </td><td> $0 . 3 8 0 { \pm } 0 . 1 5 6$ </td><td> $0 . 8 2 8 \pm 0 . 2 9 9$ </td></tr></table>

Lexicographic grammar. The lexicographic task separates exact sequence prediction from grammatical validity. The classical baseline has moderate token accuracy, $0 . 5 5 7 \pm 0 . 0 8 0$ , but perfect grammar score, meaning that it generates valid lexicographic words without reproducing the exact sampled test sequence. Quantum grammar scores increase with circuit expressivity, reaching $0 . 8 2 8 \pm 0 . 2 9 9$ for $\mathsf { q } _ { - } \mathsf { e } 3 _ { - } \mathsf { d } 3 _ { - } \mathsf { v } 2$ , which also obtains the best quantum token accuracy on this task. However, high lexicographic validity can arise from degenerate outputs such as repeated valid words. The result therefore shows that the architecture can learn formal word constraints, but also that nondeterministic grammars require diversity- or distribution-sensitive evaluation beyond validity alone.

Interpretation. Across both tasks, the main positive finding is that the quantum transformer is trainable end-to-end and can encode grammar regularities with fewer trainable parameters than the compact classical baseline. The strongest quantum variants use between 354 and 364 parameters, compared with 700 for the baseline. This is not evidence of quantum advantage, since the classical model is clearly more accurate and stable. Rather, the results show that the encoder–connector–decoder circuit design is a viable architecture for controlled autoregressive QNLP experiments, while revealing the optimization instability that future work must address.

## 6 Discussion and Limitations

The experiments support this work’s central architectural claim, namely that an autoregressive sequence model can be built from variational quantum encoder, connector and decoder blocks while retaining the external next-token interface of a transformer-style model. The positive result is not superiority over a classical transformer, but feasibility. Individual quantum runs solve the deterministic grammar and the lexicographic task shows that the model can learn word-level validity constraints that token accuracy alone does not capture.

![](images/e7a12edcce0c1eb20a11a1290077e85e49dd66ef9dc9657d5a731d08a4e39b94.jpg)  
Fig. 1. Grammar-level validity of the classical baseline and quantum transformer variants on the deterministic and lexicographic tasks. Bars show mean scores over three seeds, error bars show one standard deviation and black points indicate the individual seed outcomes.

The comparison between variants suggests that, at the tested scale, increasing variational depth inside each block is more useful than simply adding another encoder–decoder layer. However, the large standard deviations show that training remains sensitive to initialization and optimization. Good solutions appear to exist within the architecture, but the current training procedure does not find them reliably. This is consistent with broader dificulties in variational quantum optimization and should temper any interpretation of the aggregate results.

Several limitations remain. The tasks are synthetic and do not test semantic representation, compositional meaning, or natural-language generalization. The experiments use exact simulation rather than shot-based or noisy hardware execution. Finally, the direct two-qubit readout ties the present vocabulary size to four tokens. Larger vocabularies will require more readout qubits, a hybrid output map, hierarchical decoding, or another scalable readout scheme. Future work should therefore study richer encodings, more stable optimization, diversitysensitive metrics for nondeterministic grammars and execution under realistic NISQ noise and sampling constraints.

## 7 Conclusion

We introduced a NISQ-compatible quantum transformer architecture for synthetic QNLP sequence modelling. The model preserves autoregressive next-token prediction while replacing classical sequence-processing blocks with variational quantum encoder heads, connector circuits, decoder blocks and a direct twoqubit measurement readout.

On deterministic and lexicographic grammar tasks, the architecture is trainable end-to-end and learns nontrivial structure, including perfect deterministic generation in individual runs and high lexicographic validity in the strongest variant. The classical baseline remains more accurate and stable, so the results should be read as evidence for a viable architecture, not as a quantum advantage claim. Future work should extend the model to larger vocabularies and contexts, improve optimization stability and evaluate the architecture under shot-based and noisy quantum execution.

AI Assistance Disclosure. The authors used OpenAI ChatGPT for language editing. All experimental design, code, results, interpretation and final responsibility remain with the authors.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems 30, pp. 5998–6008 (2017)

2. Preskill, J.: Quantum computing in the NISQ era and beyond. Quantum 2, 79 (2018). https://doi.org/10.22331/q-2018-08-06-79

3. Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S.C., Endo, S., Fujii, K., Mc-Clean, J.R., Mitarai, K., Yuan, X., Cincio, L., Coles, P.J.: Variational quantum algorithms. Nature Reviews Physics 3, 625–644 (2021). https://doi.org/10.1038/ s42254-021-00348-9

4. Schuld, M., Killoran, N.: Quantum machine learning in feature Hilbert spaces. Physical Review Letters 122, 040504 (2019). https://doi.org/10.1103/ PhysRevLett.122.040504

5. Benedetti, M., Lloyd, E., Sack, S., Fiorentini, M.: Parameterized quantum circuits as machine learning models. Quantum Science and Technology 4(4), 043001 (2019). https://doi.org/10.1088/2058-9565/ab4eb5

6. Coecke, B., Sadrzadeh, M., Clark, S.: Mathematical foundations for a compositional distributional model of meaning. Linguistic Analysis 36(1–4), 345–384 (2010)

7. Meichanetzidis, K., Gogioso, S., de Felice, G., Chiappori, N., Toumi, A., Coecke, B.: Quantum natural language processing on near-term quantum computers. arXiv preprint arXiv:2005.04147 (2020)

8. Bergholm, V., Izaac, J., Schuld, M., Gogolin, C., Ahmed, S., Ajith, V., Alam, M.S., Alonso-Linaje, G., AkashNarayanan, B., Asadi, A., et al.: PennyLane: automatic diferentiation of hybrid quantum-classical computations. arXiv preprint arXiv:1811.04968 (2018)

9. Li, G., Zhao, X., Wang, X.: Quantum self-attention neural networks for text classification. Science China Information Sciences 67, 142501 (2024). https://doi. org/10.1007/s11432-023-3879-7

10. Guo, N., Yu, Z., Choi, M., Han, Y., Agrawal, A., Nakaji, K., Aspuru-Guzik, A., Rebentrost, P.: Quantum Transformer: Accelerating model inference via quantum linear algebra. arXiv preprint arXiv:2402.16714 (2024)

11. Khatri, N., Matos, G., Coopmans, L., Clark, S.: Quixer: A Quantum Transformer Model. arXiv preprint arXiv:2406.04305 (2024)

12. Karamlou, A., Pfafhauser, M., Wootton, J.: Quantum natural language generation on near-term devices. arXiv preprint arXiv:2211.00727 (2022)

13. Lorenz, R., Pearson, A., Meichanetzidis, K., Kartsaklis, D., Coecke, B.: QNLP in practice: Running compositional models of meaning on a quantum computer. arXiv preprint arXiv:2102.12846 (2021)

14. Kartsaklis, D., Fan, I., Yeung, R., Pearson, A., Lorenz, R., Toumi, A., de Felice, G., Meichanetzidis, K., Clark, S., Coecke, B.: lambeq: An eficient high-level Python library for quantum NLP. arXiv preprint arXiv:2110.04236 (2021)

## A Detailed Circuit Diagram

The following diagram gives the full circuit-level layout of the architecture instance used to illustrate the model structure. It is moved to the appendix to keep the main paper focused on the architectural definition and experimental results.

![](images/cc8af2bffe1553e2433a34b533957a61f47bbe5462c203e7a77ff5e030b710ba.jpg)  
Fig. 2. Detailed circuit diagram of the two-encoder, two-decoder quantum transformer instance. Embedding circuits are shown in red, individual heads in blue, encoder integration circuits in yellow, connector blocks in green and decoder blocks in orange.