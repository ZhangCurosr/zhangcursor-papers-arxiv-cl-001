# In-game Toxic Detection: Bi-directional Representations with Attention Residuals

Yuanzhe Jia

University of Sydney, Australia yjia5612@uni.sydney.edu.au

## Abstract

In-game toxic language has emerged as a critical concern in the gaming industry and community. While several frameworks and models for online game toxicity analysis have been proposed, detecting toxicity in player chat utterances remains a formidable challenge: stemming not only from the extremely short length of such utterances but also from the heavy reliance on game slang, abbreviations, and domainspecific jargon, which generic language models are poorly suited to recognize. This paper presents a shared task for in-game toxic language detection built upon real-world ingame chat data, and proposes the best-preforming model for the toxic language slot filling: Bi-directional Representations with Attention Residuals (BRAR). Experimental results demonstrate that BRAR effectively captures the global context and outperforms the existing baselines on slot filling. The relevant code is publicly available on GitHub<sup>1</sup>.

## Introduction

Toxic behavior has become a critical concern in online gaming, yet detection is complicated by the distinctive nature of in-game chat: unlike social media or news, it is substantially shorter because players type while playing, with longer utterances confined to pre- or post-game discussions. This brevity, compounded by pervasive game slang, abbreviations, and domain-specific jargon that generic language models often fail to capture, renders slot-level analysis indispensable for detecting in-game toxicity. Slot filling, the core technique underlying such token-level detection, is typically cast as sequence labeling and has progressed from feature-engineered statistical models such as CRF to deep learning approaches, with Mesnil first demonstrating that the bi-directional RNN with CRF decoders can improve the performance (Mesnil et al. 2014). However, RNNs suffer from vanishing/exploding gradients and fail to capture long-term dependencies explicitly; LSTMs and GRUs mitigate this via gating but still lack explicit dependency modeling and parallel processing. The attention mechanism was therefore adopted, and several sequence-to-sequence models with the attention layer have been applied to slot filling, including encoder–decoder enhancement (Liu and Lane

2016), position-aware attention (Zhang et al. 2017), joint delexicalized generation and label prediction (Shin, Yoo, and Lee 2018), and pointer-network-based slot value prediction (Zhao and Feng 2018). Nevertheless, few approaches integrate BiLSTM, Attention, and CRF to capture adjacentword interactions, highlight key words, and exploit label dependencies. To address the gap, this paper describes the established slot (token)-based in-game toxic language detection shared task and presents the best-performing model with its novel components.

## Shared Task and Dataset

We established a shared task competition<sup>2</sup> for sequence token labeling, aiming to identify the semantic category of each slot in chat utterances from CONDA (Weld et al. 2021), a dataset comprising 44,869 utterances derived from chat logs of 1,921 Dota 2 matches and annotated with six distinct slot labels—T (Toxicity), C (Character), D (Dota-specific), S (Game Slang), P (Pronoun) and O (Other)—to facilitate a deeper understanding of game context. Given the informal and noisy nature of in-game chat, the task demanded models capable of handling lexical variants and domain-specific terminology. Hosted on Kaggle, the competition attracted 312 teams, each permitted 20 submissions per day, culminating in 3,646 submissions over four weeks, with evaluation based on overall micro-F1 excluding the O tag. To ensure reproducibility, a standard data split and evaluation script were released, and participants were encouraged to report both overall and per-label F1 scores. Most participants preprocessed the provided data by tokenizing utterances into slots using spaces, as CONDA provides cleaned utterances with slot labels, while some employed data augmentation to enlarge their training sets. For input embeddings, participants leveraged combinations of syntactic, semantic, and domainrelated representations: syntactic embeddings typically encoded POS tags and/or dependency parsing results from SpaCy, whereas semantic embeddings either adopted pretrained GloVe/FastText vectors or were trained from scratch using FastText/Word2Vec on the provided game chat corpus. Diverse sequence labeling architectures were explored, and we subsequently present the methodology and experimental results of the best-performing model.

![](images/7267b3b3ed0bbdd239fd1dbd64d7bf9f1f95f184e16e28d61bb1c7e40500930d.jpg)  
Figure 1: The figure illustrates the overall architecture of BRAR, in which the dotted arrows denote the attention residuals and the green boxes (labeled “LF”) present the label forcing technique.

## Methodology

The proposed model is a combination of BiLSTM cells, attention residuals, the label forcing technique, and CRF decoders. Since the model uses the global information extracted from the attention mechanism as residuals to supplement bi-directional features, it is named Bi-directional Representations with Attention Residuals (BRAR). The overall architecture is shown in Figure 1. The input example, “gg SEPA Report my team”, is divided into the sequence $\boldsymbol { x } ~ = ~ \left( x _ { 1 } , x _ { 2 } , \ldots , x _ { t } \right)$ where t denotes the number of tokens. The output of the model is the corresponding slot labels $y = ( y _ { 1 } , y _ { 2 } , \dots , y _ { t } )$ with c unique values.

## BiLSTM

The BiLSTM layer performs feature extraction on the input data in sequence, and the hidden state obtained at each time step is then passed to the attention layer. Assume the hidden size of LSTM is λ, the hidden state $\bar { h _ { i } } \in \mathbb { R } ^ { \lambda }$ at time step i is represented as the abbreviation below:

$$
h _ { i } = L S T M ( x _ { i } , h _ { i - 1 } )\tag{1}
$$

And the bi-directional hidden state $h _ { i } \in \mathbb { R } ^ { 2 \lambda }$ that involves the forward hidden state $h _ { i } ^ { f o r w a r d }$ and the backward hidden state h<sup>backward</sup><sub>i</sub> is depicted:

$$
h _ { i } = [ h _ { i } ^ { f o r w a r d } , h _ { i } ^ { b a c k w a r d } ]\tag{2}
$$

## Attention Residual

The attention layer aims to understand the global information and find the main ideas of the input utterance. The attention $a \in \mathbb { R } ^ { 2 \lambda }$ is calculated in the following equation where $h _ { t } ~ \in ~ \mathbb { R } ^ { 2 \lambda }$ is the last hidden state of the BiLSTM layer, $\bar { W } _ { a } \in \mathbb { R } ^ { 2 \lambda \times t }$ is a trainable weight matrix and $H \in \mathbb { R } ^ { t \times 2 \lambda }$ is output the hidden state of BiLSTM:

$$
a = s o f t m a x ( h _ { t } \cdot W _ { a } \cdot H )\tag{3}
$$

![](images/0edcd5c32d57fb52f55d26334aa41169fb0b69c246b31fdc41f3794fb2b2d86a.jpg)  
Figure 2: The figure illustrates the process by which the parameter matrices in BRAR undergo a series of dimensional transformations and ultimately yield the emission scores for the CRF layer.

As the global information is not always helpful for understating each input token, it is treated as the residual of the token-level representation to form the feature $f _ { i } \in \mathbb { R } ^ { t \times c }$ while α is a trainable parameter for scaling and $W _ { f } \in \mathsf { \Gamma }$ $\mathbb { R } ^ { 2 \lambda \times c }$ is a trainable weight matrix for dimension transformation. When the global information is beneficial to the token-level interpretation, it will speed up the convergence; otherwise, it will not reduce the prediction performance.

$$
f _ { i } = W _ { f } ( H + \alpha \times a _ { i } )\tag{4}
$$

## Label Forcing

The feature representation is enhanced at the label forcing layer to form the emission scores, which will be passed to the CRF layer to predict the tag of each token. As both training and test sets contain a large number of identical words, the classifications of these words are highly likely to be the same in a specific domain. For example, “gg” stands for “good game”. If all the words “gg” in the training set are classified as “S” (Game Slang), then the word in the test set will have a high probability of being classified as “S”. That is to say, if the probability information of the correspondence between words and labels in the training set can be learned, it will be of great help in predicting the results of the test set. Therefore, the proposed model uses the label forcing technique: the label distribution probability of each token in the training corpus is calculated and normalized by the following equation, where $C _ { i j }$ is the frequency of that the token i is annotated as the label j:

$$
p _ { i } = C _ { i j } / \sum _ { j } C _ { i j }\tag{5}
$$

After that, the label distribution probability $p _ { i }$ will be added element-wise along with the feature representation $f _ { i }$ to form the emission scores $\boldsymbol { e } _ { i } \in \mathbb { R } ^ { t \times c }$ for the CRF layer (Ref. Figure 2):

$$
e _ { i } = f _ { i } + p _ { i }\tag{6}
$$

<table><tr><td>Number of Stacks</td><td>F1</td><td>F1(T)</td><td>F1(S)</td><td>F1(D)</td></tr><tr><td>1 layer</td><td>99.9</td><td>98.6</td><td>99.4</td><td>98.1</td></tr><tr><td>2 layers</td><td>99.6</td><td>98.1</td><td>99.3</td><td>96.4</td></tr><tr><td>3 layers</td><td>98.6</td><td>96.5</td><td>98.3</td><td>81.8</td></tr></table>

Table 1: Ablation study with different stacks (%).
<table><tr><td>Model</td><td>F1</td><td>F1(T)</td><td>F1(S)</td><td>F1(D)</td></tr><tr><td>RNN-NLU (2016)</td><td>97.0</td><td>93.1</td><td>93.0</td><td>71.8</td></tr><tr><td>Slot-gated (2018)</td><td>99.1</td><td>97.8</td><td>98.2</td><td>95.2</td></tr><tr><td>Inter-BiLSTM (2018)</td><td>86.5</td><td>87.1</td><td>86.9</td><td>78.8</td></tr><tr><td>Capsule NN (2019)</td><td>99.1</td><td>97.5</td><td>98.2</td><td>94.9</td></tr><tr><td>Joint BERT (2019)</td><td>98.9</td><td>97.2</td><td>97.9</td><td>91.4</td></tr><tr><td>BRAR (our model)</td><td>99.9</td><td>98.6</td><td>99.4</td><td>98.1</td></tr></table>

Table 2: Comparison with baselines (%).

## CRF

Given that slot label predictions are inherently interdependent, modeling the transitional dependencies among neighboring labels is beneficial for producing coherent tag sequences. Accordingly, a CRF layer is placed atop the proposed model to jointly decode the optimal chain of labels for the utterance. This design enables the model to enforce valid tag transitions and prevents locally optimal but globally inconsistent predictions.

## Experiment

The proposed model adopts FastText-50 as input embeddings, with a hidden size of 5 per direction in the BiLSTM layer and the number of layers fixed at 1. The SGD optimizer is employed with a learning rate of 0.1 and a weight decay of 1e-4, and training proceeds for 2 epochs with a batch size of 1, following the CONDA setting. Evaluation metrics include the overall micro-F1 and per-slot F1 on the test set, excluding the O tag when computing the overall F1, and five baselines are taken from the CONDA paper. We first conducted an ablation study on the number of stacked BiL-STM layers. As presented in Table 1, increasing the number of layers leads to lower F1 scores, although the differences among the three configurations are not statistically significant. This suggests that deeper architectures tend to overfit the relatively short and noisy in-game chat utterances. Consequently, we use a single BiLSTM layer owing to its lower parameter count and computational efficiency. In comparison with the existing baselines, Table 2 shows that BRAR achieves superior F1 scores across most labels, especially for the T, S, and D labels, attributable to the attention residuals and the label forcing technique that enhance the capture of global context and high-frequency label patterns.

## Conclusion

In this paper, we established a shared task for slot (token)- based in-game toxic language detection, as well as the best-performing model in the shared task, which integrates bi-directional representations, attention residuals, the label forcing technique, and CRF decoders. Experiments indicate that the proposed model is more effective in capturing global information between the semantic components on slot filling than the existing baselines.

## Declaration

This paper constitutes an extended version of the original paper (Jia et al. 2023), primarily incorporating the related work, and providing a detailed elaboration of the methodology of the proposed model.

## References

Bing, L.; and Ian, L. 2016. Attention-based recurrent neural network models for joint intent detection and slot filling. In Interspeech 2016, 685–689.

Chenwei, Z.; Yaliang, L.; Nan, D.; Wei, F.; and S, P., Yu. 2019. Joint slot filling and intent detection via capsule neural networks. In Proceedings ofthe 57th Annual Meeting ofthe ACL, 5259–5267.

Chih-Wen, G.; Guang, G.; Yun-Kai, H.; Chih-Li, H.; Tsung-Chieh, C.; Keng-Wei, H.; and Yun-Nung, C. 2018. Slotgated modeling for joint slot filling and intent prediction. In NAACL-HLT 2018, 753–757.

Jia, Y.; Wu, W.; Cao, F.; and Han, S. C. 2023. In-game toxic language detection: Shared task and attention residuals. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 16238–16239.

Liu, B.; and Lane, I. 2016. Attention-based recurrent neural network models for joint intent detection and slot filling. arXiv preprint arXiv:1609.01454.

Mesnil, G.; Dauphin, Y.; Yao, K.; Bengio, Y.; Deng, L.; Hakkani-Tur, D.; He, X.; Heck, L.; Tur, G.; Yu, D.; et al. 2014. Using recurrent neural networks for slot filling in spoken language understanding. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 23: 530–539.

Shin, Y.; Yoo, K. M.; and Lee, S.-g. 2018. Slot Filling with Delexicalized Sentence Generation. In INTERSPEECH, 2082–2086.

Weld, H.; Huang, G.; Lee, J.; Zhang, T.; Wang, K.; Guo, X.; Long, S.; Poon, J.; and Han, C. 2021. CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection. In Findings of the Association for Computational Linguistics: ACL 2021, 2406–2416.

Yu, W.; Yilin, S.; and Hongxia, J. 2018. A bi-model based rnn semantic frame parsing model for intent detection and slot filling. In NAACL-HLT 2018.

Zhang, Y.; Zhong, V.; Chen, D.; Angeli, G.; and Manning, C. D. 2017. Position-aware attention and supervised data improve slot filling. In Conference on Empirical Methods in Natural Language Processing.

Zhao, L.; and Feng, Z. 2018. Improving slot filling in spoken language understanding with joint pointer and attention. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 426–431.