# Size Matters: Foundation Model for Czech HTML documents

Martin Dvořák <sup>⋆</sup>, Vít Tlustoš (B) <sup>⋆</sup>, Artyom Voronin <sup>⋆</sup>, Martin Habrovec , Kateřina Podlesná , Barbora Rišová , and Josef Vonášek

Seznam.cz, Prague, Czech Republic vit.tlustos@firma.seznam.cz

Abstract. Creating universal, high-quality representations of web documents in high-trafic industrial environments requires models that are both performant and economic. Existing approaches, however, often depend on large models, overlook the structural information inherent in HTML, or are constrained by short context windows, limiting their ability to process real-world web pages. We present HTML-LM, a compact foundation model with 154 million parameters that addresses these limitations through HTML-aware training and a ModernBERT-based architecture. It was trained on 100 million web documents using multiple objectives, including masked language modeling, bag-of-words prediction, and contrastive distillation from large language models. Consequently, HTML-LM sets a new state-of-the-art for classification and regression applications in the Czech Internet domain, surpassing both larger encoders and small-sized LLMs. The model is deployed in production, processing thousands of web documents per second, and released to the community under the CC BY-NC 4.0<sup>1</sup> license.

https://huggingface.co/Seznam/html-lm

Keywords: HTML foundation model · web document representation · web document understanding

## 1 Introduction

In information retrieval systems, categorical metadata and document-level signals play a critical role in ensuring retrieval quality from the Internet. Signals such as document type [22], spam likelihood [10], or the presence of adult content are essential for maintaining clean search indexes and improving search ranking and user satisfaction. However, at scale, incorporating these signals into real-time retrieval pipelines requires representations that are both computationally eficient and semantically expressive. An efective strategy is to encode each document into a compact embedding that serves as a shared representation for multiple downstream models.

Prior research into HTML document representation has evolved along several trajectories. Structure-aware models based on XPath-like encodings, such as MarkupLM [8], Structor [24], and DOM-LM [3], treat HTML tags and DOM structure as first-class signals. Other architectures, notably WebFormer [18], create complex attention patterns between HTML and text tokens, thereby training the model to capture HTML structure. However, such models are restricted by token windows (typically 512 tokens), which limits their capacity to encode entire web pages (see Section 3.3). Contemporary state-of-the-art text encoders such as Jina’s jina-embeddings-v3 [17] and OpenAI’s text-embeddings-3 [13] provide strong semantic representations, yet their operational costs make inference over large, continuously evolving web corpora expensive. Large language models (LLMs) have also demonstrated promising results on tasks including HTML understanding [7]. Nevertheless, their scale makes them impractical for large-scale deployment.

To the best of our knowledge, no existing model explicitly targets the problem of HTML document embedding while simultaneously emphasizing computational eficiency and the large context lengths required to represent real-world web pages. Furthermore, existing approaches predominantly focus on English, whereas our work primarily targets the Czech language, reflecting the needs of Seznam.cz as a Czech-based search engine. In this work, we present HTML-LM, a model that eficiently compresses web page content into high-quality embeddings suitable for a wide range of classification and regression tasks.

## 2 Methodology

## 2.1 Exploratory Analysis and Design Choices

We conducted a comprehensive exploratory analysis to guide our decisions. First, we developed a suite of benchmarks (see Section 3.2) to enable systematic comparisons across models. Using these benchmarks, we evaluated several state-ofthe-art models without any task-specific fine-tuning.

This evaluation identified Qwen3-Embedding-8B [23] as a strong performer, supporting the feasibility of universal HTML document representations. Further analyses revealed two critical factors for efective HTML processing: (i) a minimum context window of 2048 tokens (see Section 3.3), and (ii) explicit preservation of HTML structure (see Section 3.4).

## 2.2 Data

We trained our model on a large corpus of 100 million HTML documents. The dataset comprised 53% Czech domains (.cz), 34% primarily English-language domains (.com, .org, .net), 9% other European domains (.sk, .de, .eu, .pl, .it, .at), and the remaining 4% from less common domains. The dataset was sampled from our internal web database, which stores crawled pages along with related metadata, including text length, domain, and document cluster. We used these metadata to systematically design and refine a sampling strategy. After extensive experimentation, our sampling procedure produced a dataset with text lengths uniformly distributed, a limit of 1000 documents per domain, and clusters with high entropy prioritized.

Pre-processing We adopted an aggressive preprocessing strategy similar to [6]. When tokenized, preprocessed HTML typically retains fewer than 5% of the original tokens while preserving most of the information needed for accurate downstream task processing. The algorithm operates as follows:

1. HTML Parsing: The HTML document is parsed using BeautifulSoup4 with the lxml backend, producing a Document Object Model (DOM) tree.

2. DOM Pruning: During a depth-first DOM traversal, the tree is processed as follows:

– Removal: We remove all subtrees without plain-text content as well as all <script>, <style>, and comment nodes.

Retention: All textual content is preserved, including tags around it. In addition, the following tags are explicitly retained even when they do not contain text, as they encode relevant structural or semantic information: a, address, audio, br, button, col, embed, figure, footer, form, frame, header, hr, iframe, img, input, label, menu, meter, nav, option, output, picture, progress, search, select, td, th, tr, textarea, video, i, svg.

– Attribute Stripping: Finally, attributes are removed from each retained node.

3. DOM Compression: Nodes with exactly one child are replaced by their child, reducing unnecessary hierarchy.

4. Whitespace Normalization: Multiple consecutive whitespace characters are collapsed into a single space.

Tokenization To improve tokenization eficiency, we created a custom HTMLoptimized WordPiece tokenizer [21] with a 57K-token vocabulary and a minimum token frequency of 10, balancing sequence length (benefiting from larger vocabularies) against model size (the embedding matrix size). Using our training corpus, we identified the 100 most frequent HTML tags and assigned dedicated tokens for their opening and closing forms. The number was selected empirically; including more than 100 tags provided no gains, while keeping only the top 50 tags worsened the tokenization performance. The tokenizer was trained from scratch on 5M randomly sampled documents from the model training corpus (see Section 2.2), as larger subsets did not improve training without providing measurable benefits.

## 2.3 Model

Our model is based on the ModernBERT [19] architecture, which supports context windows of 8192 tokens and allows for eficient processing of long documents. The proposed HTML-LM model consists of 22 layers and 12 attention heads, employs a hidden size of 768, and contains a total of 154 million parameters. Because we replaced the tokenizer and modified key hyperparameters, the model was trained from scratch for a single complete pass over the training dataset using the Adam optimizer with a learning rate of $2 . 5 \cdot 1 0 ^ { - 5 }$ . Training employed a trapezoidal linear learning-rate schedule with warm-up and cooldown phases covering 10% and 20% of the total training steps, respectively. The maximum sequence length (during training) was limited to 4096 tokens, and the batch size was set to 32 per GPU, yielding a global batch size of 256 distributed across 8 NVIDIA H100 GPUs.

## 2.4 Training Objectives

We trained our model in a multi-task setting with objectives, including Masked Language Modeling [4], the Bag-of-Word Prediction [11], and contrastive distillation of LLM embeddings [12].

Projection Heads To make loss functions computable, embeddings from the model’s hidden dimension D are projected into loss-specific target spaces of dimension T.

– Language Modeling (LM) and Bag-of-Words (BOW) Heads map embeddings from D to the full vocabulary space |V |. Instead of introducing a new projection layer, we use the transpose of the model’s embedding matrix [14]. To further improve eficiency, we use the Cut Cross-Entropy (CCE) [20].

– Distillation Head projects embeddings from D to the teacher space T via a low-rank factorization (Equation 1). The rank R reduces the number of additional parameters while approximating the full $D \times T$ projection. This head is discarded after training.

$$
W ^ { D \times T } \approx W ^ { D \times R } \times W ^ { R \times T } \quad \mathrm { w h e r e } \quad R < D < T\tag{1}
$$

Masked Language Modeling (MLM) Following standard practice, we adopt MLM as a training objective. By masking both textual and HTML tokens, the model learns to capture both semantic content and structural information from the DOM, while reinforcing token-level representations.

Bag-of-Words Prediction (BOW) BOW operates on masked inputs like MLM, but instead of reconstructing each token from its local context, it recovers all tokens from the [CLS] representation, encouraging a global, document-level understanding.

LLM Distillation To incorporate knowledge from substantially larger teacher models, we employ Teacher-Space Weighted Contrastive Distillation, using Qwen3- Embedding-8B [23] and SeLLMa 8B (internal LLM model based on Llama $3 . 1 ^ { 2 } \ [ 5 ]$ and fine-tuned for the Czech language) [15] [16] as teachers. Unlike standard inbatch contrastive losses such as InfoNCE [12] and SimCLR [2], which treat all negatives equally, our method weights each negative pair based on teacher guidance. Negatives considered similar by the teacher are penalized less, allowing closer representations. Our implementation builds on SoftCSE [25] but introduces a diferent teacher guidance scheme. Formally, for a batch B, we define the distillation loss in Equation 2.

$$
\mathcal { L } _ { \mathrm { d i s t i l } } = \mathbb { E } _ { i \in B } \left[ - \log \frac { \exp \left( s _ { i , i } / \tau \right) } { \exp \left( s _ { i , i } / \tau \right) + \sum _ { j \in B , j \ne i } w _ { i , j } \exp \left( s _ { i , j } / \tau \right) } \right]\tag{2}
$$

Here, $s _ { i , j }$ denotes the cosine similarity between the student’s embedding of document i and the teacher’s embedding of document $j .$ . The teacher guidance weight $\begin{array} { r } { w _ { i , j } = \frac { 1 - \tilde { s } _ { i , j } } { 2 } \in [ 0 , 1 ] } \end{array}$ is based on $\tilde { s } _ { i , j ; \ l }$ which represents the cosine similarity between the teacher’s representation of documents i and $j .$ Finally, the temperature $\tau$ controls the sharpness of the distribution. We found $\tau = 0 . 1$ to perform the best.

Aggregation When using multiple loss functions during training, these losses must be unified into a single objective ${ \mathcal { L } } .$

$$
\mathcal { L } = \sum \lambda _ { i } \cdot \alpha _ { i } \cdot \frac { \mathcal { L } _ { i } } { \mathcal { L } _ { i } ^ { i n i t i a l } } \quad \alpha _ { i } \sim \mathcal { U } ( 0 , 1 )\tag{3}
$$

Because individual losses difer in scale, we first estimate the magnitude of each loss from the initial batch, $\mathcal { L } _ { i } ^ { \mathrm { i n i t i a l } }$ , and then normalize it so that all losses afect the overall objective to a similar extent. According to Equation $^ { 3 , }$ the total loss $\mathcal { L }$ is formulated as a weighted sum of the individual normalized loss terms. Additionally, each loss term is scaled by a fixed baseline weight $\lambda _ { i }$ $( \lambda _ { \mathrm { M L M } } = 0 . 1 , \lambda _ { \mathrm { B O W } } = 1 , \lambda _ { \mathrm { d i s t i l } } = 1 )$ , which specifies the relative importance of each objective, and a stochastic coeficient $\alpha _ { i }$ drawn uniformly from $\mathcal { U } ( 0 , 1 )$ at every step. This strategy helps the model prioritize diferent training aspects at each step, thereby improving generalization. [9]

## 3 Evaluation

## 3.1 Tokenizer Performance

To evaluate the proposed tokenizer, we measured the percentage of documents that fit entirely within a given context window after tokenization. This evaluation was performed on a random sample of 100K documents from the model training corpus (see Section 2.2). As shown in Table 1, our tokenizer demonstrates robust performance across all context windows, outperforming all competitors except MarkupLM and SeLLMa 8B. While MarkupLM delivers slightly better performance, it depends on extracting structured information—specifically the XPath of each HTML node—from the DOM, which is computationally expensive. Similarly, SeLLMa’s superior performance is likely driven by its large 128K vocabulary. Consequently, neither is suitable for our specific setting. Overall, among usable tokenizers, our approach achieves the best coverage across all evaluated context window sizes.

<table><tr><td>Tokenizer</td><td>Size</td><td>≤ 512|</td><td>4096</td><td>8192</td><td>Util.</td></tr><tr><td>MarkupLM** tiktoken cl100k_base*</td><td>50K</td><td>25.8 10.8</td><td>88.7 80.6</td><td>96.3 93.5</td><td>99.5 84.1</td></tr><tr><td>ModernBERT</td><td>100K 50K</td><td>10.0</td><td>79.3</td><td>92.9</td><td>97.3</td></tr><tr><td>jina-embeddings-v3</td><td>250K</td><td>9.8</td><td>80.7</td><td>93.6</td><td>77.7</td></tr><tr><td>Qwen3-Embedding-8B</td><td>151K</td><td>10.4</td><td>80.2</td><td>93.4</td><td>78.6</td></tr><tr><td>SeLLMa 8B</td><td>128K</td><td>29.2</td><td>92.2</td><td>97.5</td><td>86.2</td></tr><tr><td></td><td>57K</td><td>7.9</td><td></td><td></td><td></td></tr><tr><td>RetroMAE [1]</td><td></td><td></td><td>77.2</td><td>92.1</td><td>97.0</td></tr><tr><td>HTML-LM (ours)</td><td>57K</td><td>10.3</td><td>83.7</td><td>94.8</td><td>98.9</td></tr></table>

Table 1. Tokenizer performance, measured as the percentage of documents that, when tokenized, fully fit within context windows of varying lengths (≤ N). The Util. column represents vocabulary utilization, while the Size column indicates the vocabulary size. Used by the OpenAI text-embeddings-3.  
∗∗ Does not tokenize HTML tags.

## 3.2 Evaluation Methodology

The model performance is assessed across multiple downstream applications, with an emphasis on documents from the Czech domain. During evaluation, the model under investigation remains completely frozen, and only a lightweight, task-specific MLP head is trained on its embeddings. Depending on the task, each head introduces approximately 0.3–1M additional parameters. As presented in Table 2, our model, although significantly smaller than its main competitors, outperforms all other models.

## Downstream Applications

1. Article Type (multi-class classification, metric: F1 macro)

– Evaluates the model’s ability to classify article type. – Classes: Not an article, Tabloid, Journalism, Hobby, Sport, News – local, News – domestic & international.

2. Curlie (multi-class classification, metric: Accuracy) – Evaluates the model’s ability to classify Czech webpages according to the top-level Curlie categories<sup>3</sup>.

<table><tr><td rowspan=2 colspan=1>Modelmetric</td><td rowspan=2 colspan=1>Params</td><td rowspan=2 colspan=1>Article TypeF1 macro ↑</td><td rowspan=2 colspan=1>CurlieAccuracy ↑</td><td rowspan=1 colspan=1>Porn</td><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>Web Spam</td><td rowspan=1 colspan=1>Aggregated</td></tr><tr><td rowspan=1 colspan=1>F1 macro ↑</td><td rowspan=1 colspan=1>F1 macro ↑</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>NMM↑</td></tr><tr><td rowspan=8 colspan=1>randomMarkupLM baseModernBERT baseOpenAI text-embeddings-3 smalljina-embeddings-v3 baseQwen3-EmbeddingSeLLMa 8BHTML-LM base</td><td rowspan=4 colspan=1>0135M149M?</td><td rowspan=2 colspan=1>0.06910.6853</td><td rowspan=2 colspan=1>0.03280.2198</td><td rowspan=1 colspan=1>0.3322</td><td rowspan=1 colspan=1>0.1914</td><td rowspan=1 colspan=1>0.5619</td><td rowspan=1 colspan=1>0.0000</td></tr><tr><td rowspan=1 colspan=1>0.5684</td><td rowspan=1 colspan=1>0.7943</td><td rowspan=1 colspan=1>0.3120</td><td rowspan=1 colspan=1>0.4799</td></tr><tr><td rowspan=2 colspan=1>0.70850.7650</td><td rowspan=1 colspan=1>0.3177</td><td rowspan=1 colspan=1>0.4864</td><td rowspan=1 colspan=1>0.8128</td><td rowspan=1 colspan=1>0.3166</td><td rowspan=1 colspan=1>0.4835</td></tr><tr><td rowspan=1 colspan=1>0.7406</td><td rowspan=1 colspan=1>0.6348</td><td rowspan=1 colspan=1>0.8513</td><td rowspan=1 colspan=1>0.2678</td><td rowspan=1 colspan=1>0.6544</td></tr><tr><td rowspan=1 colspan=1>570M</td><td rowspan=1 colspan=1>0.7856</td><td rowspan=1 colspan=1>0.7346</td><td rowspan=1 colspan=1>0.6250</td><td rowspan=1 colspan=1>0.8753</td><td rowspan=1 colspan=1>0.3070</td><td rowspan=1 colspan=1>0.6466</td></tr><tr><td rowspan=1 colspan=1>8B</td><td rowspan=1 colspan=1>0.7732</td><td rowspan=1 colspan=1>0.7536</td><td rowspan=1 colspan=1>0.6442</td><td rowspan=1 colspan=1>0.8937</td><td rowspan=1 colspan=1>0.3102</td><td rowspan=1 colspan=1>0.6571</td></tr><tr><td rowspan=2 colspan=1>8B154M</td><td rowspan=1 colspan=1>0.7568</td><td rowspan=1 colspan=1>0.7406</td><td rowspan=1 colspan=1>0.5592</td><td rowspan=1 colspan=1>0.7933</td><td rowspan=1 colspan=1>0.2827</td><td rowspan=1 colspan=1>0.6103</td></tr><tr><td rowspan=1 colspan=1>0.7978</td><td rowspan=1 colspan=1>0.7577</td><td rowspan=1 colspan=1>0.6521</td><td rowspan=1 colspan=1>0.9188</td><td rowspan=1 colspan=1>0.2306</td><td rowspan=1 colspan=1>0.7001</td></tr></table>

Table 2. Comparison of model performance across all downstream tasks. All models were evaluated using HTML-preserving inputs and an 8192-token context window, except for MarkupLM, which supports only 512 tokens, and the random model, which receives no input. The random model’s scores were derived from heads trained on random embeddings.

Categories: Arts, Business, Computers, Health, Home, News, Science, Sports, Shopping, Kids and Teens.

3. Porn (multi-class classification, metric: F1 macro)

– Evaluates the model’s ability to classify explicit content.   
– Classes: Safe, Adult, Porn.

4. Product (multi-class classification, metric: F1 macro)

– Evaluates the model’s ability to identify product-related pages.

– Classes: E-shop product list, E-shop product detail, Other.

5. Web Spam (regression, metric: RMSE)

– Measures how well the model can predict the amount of spam in the page.

– Output: continuous score representing the spam amount.

Normalized Metric Mean (NMM) We use the Normalized Metric Mean (NMM) to evaluate performance across multiple downstream applications. This metric represents the average improvement of a model over a baseline R. For each task t, we calculate a contribution $C _ { t }$ based on the model performance $M _ { t }$ relative to the baseline $R _ { t }$ , where $\begin{array} { r } { C _ { t } = \frac { M _ { t } ^ { - } - R _ { t } } { 1 - R _ { t } } } \end{array}$ if $M _ { t }$ is maximized, and $\begin{array} { r } { C _ { t } = 1 - \frac { M _ { t } } { R _ { t } } } \end{array}$ if $M _ { t }$ is minimized. These contributions are then averaged across all $T$ tasks, yielding a single aggregated metric that facilitates easy comparison of multiple models. Using a random model as the baseline, the NMM quantifies the extent to which our approach outperforms chance.

## 3.3 Long Context

Following the methodology outlined in Section 3.2, we examined the context length required for efective HTML document processing by measuring performance on inputs truncated to 512, 2048, 4096, and 8192 tokens. As shown in Table 3, Qwen3-Embedding-8B demonstrates consistent performance improvements up to 4096 tokens, after which performance stabilizes. Our HTML-LM model shows a similar trend, delivering strong performance at 2048 tokens, with continued improvement as the context length increases up to the model’s maximum of 8192 tokens.

<table><tr><td>Model metric</td><td>≤ 512 NMM ↑</td><td>≤ 2048 NMM ↑</td><td>≤ 4096 NMM ↑</td><td>≤ 8192 NMM ↑</td></tr><tr><td>Qwen3-Embedding (8B) HTML-LM base (154M)</td><td>0.6327 0.6807</td><td>0.6486 0.6952</td><td>0.6581 0.6954</td><td>0.6571 0.7001</td></tr></table>

Table 3. Model performance measured on HTML-preserving inputs truncated to varying context lengths (≤ N).

## 3.4 HTML vs. Plain Text

Following the methodology outlined in Section 3.2, we assessed the efect of preserving HTML markup on model performance. We compared models on content presented in two forms: with HTML tags preserved and with HTML tags removed (leaving only plain text). As shown in Table 4, preserving HTML tags improves performance for the Qwen3-Embedding-8B model, even though the model was not explicitly trained on HTML. Motivated by this finding, we trained the proposed HTML-LM model on HTML-preserving inputs and, for comparison, trained an identical model on plain-text inputs. This direct comparison—using the same architecture and training setup, difering only in input format—demonstrates that preserving HTML improves performance. In the HTML-LM setup, preserving HTML improves NMM by 0.0064, a gain roughly comparable to increasing the model size from 75M to 154M parameters.

<table><tr><td rowspan=1 colspan=1>Modelmetric</td><td rowspan=1 colspan=1>HTMLNMM ↑</td><td rowspan=1 colspan=1>TextNMM ↑</td></tr><tr><td rowspan=2 colspan=1>Qwen3-Embedding (8B)HTML-LM base (154M)</td><td rowspan=2 colspan=1>0.65710.7001</td><td rowspan=1 colspan=1>0.6504</td></tr><tr><td rowspan=1 colspan=1>0.6937</td></tr></table>

Table 4. Comparison of model performance on inputs with HTML preserved (HTML) versus inputs with HTML removed (Text). Models were evaluated using an 8192-token context.

## 4 Conclusion

We presented HTML-LM, a compact, HTML-aware foundation model designed to generate versatile, high-quality representations of web documents, with a focus on the Czech Internet domain. By leveraging a HTML-informed training, an aggressive preprocessing pipeline, and a HTML-optimized tokenizer, HTML-LM produces generalizable embeddings while remaining computationally eficient. Despite having only 154 million parameters, it achieves state-of-the-art results across multiple Czech classification and regression benchmarks, outperforming both larger embedding models and small-sized LLMs. HTML-LM has also been deployed at scale in production, processing thousands of documents per second, demonstrating strong performance and a real-world industrial impact.

Acknowledgments. This work was carried out as part of the project HTML-LM, funded by Seznam.cz. This preprint has not undergone peer review (when applicable) or any post-submission improvements or corrections. The Version of Record of this contribution is published in Text, Speech, and Dialogue (TSD 2026), and is available online at https://doi.org/10.1007/978-3-032-37249-9\_12.

Disclosure of Interests. All authors are employees of Seznam.cz. The study was conducted within the HTML-LM project, and its outcomes are used in the company’s production systems.

## References

1. Bednář, J., Náplava, J., Barančíková, P., Lisick\`y, O.: Some like it small: Czech semantic embedding models for industry applications. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 22734–22742 (2024). https://do i.org/10.1609/aaai.v38i21.30307

2. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International conference on machine learning. pp. 1597–1607. PmLR (2020). https://doi.org/10.48550/arXiv.200 2.05709

3. Deng, X., Shiralkar, P., Lockard, C., Huang, B., Sun, H.: Dom-lm: Learning generalizable representations for html documents. arXiv preprint arXiv:2201.10608 (2022). https://doi.org/10.48550/arXiv.2201.10608

4. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). pp. 4171–4186 (2019). https://doi.org/10.18653/v1/N19-1423

5. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024). https://doi.org/10.48550/arXiv.240 7.21783

6. Guo, Y., Ma, Z., Mao, J., Qian, H., Zhang, X., Jiang, H., Cao, Z., Dou, Z.: Webformer: Pre-training with web pages for information retrieval. In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 1502–1512 (2022). https://doi.org/10.1145/347749 5.3532086

7. Gur, I., Nachum, O., Miao, Y., Safdari, M., Huang, A., Chowdhery, A., Narang, S., Fiedel, N., Faust, A.: Understanding html with large language models. arxiv 2022. arXiv preprint arXiv:2210.03945 (2022). https://doi.org/10.48550/arXiv .2210.03945

8. Li, J., Xu, Y., Cui, L., Wei, F.: Markuplm: Pre-training of text and markup language for visually rich document understanding. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 6078–6087 (2022). https://doi.org/10.18653/v1/2022.acl-long.420

9. Lin, B., Ye, F., Zhang, Y., Tsang, I.W.: Reasonable efectiveness of random weighting: A litmus test for multi-task learning. arXiv preprint arXiv:2111.10603 (2021). https://doi.org/10.48550/arXiv.2111.10603

10. Lueg, C.P.: From spam filtering to information retrieval and back: seeking conceptual foundations for spam filtering. Proceedings of the American society for information science and technology 42(1) (2005). https://doi.org/10.1002/me et.14504201146

11. Ma, G., Wu, X., Lin, Z., Hu, S.: Drop your decoder: Pre-training with bag-of-word prediction for dense passage retrieval. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 1818–1827 (2024). https://doi.org/10.1145/3626772.3657792

12. Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018). https://doi.org/10.48550 /arXiv.1807.03748

13. OpenAI: New embedding models and API updates (1 2024), https://openai.com /index/new-embedding-models-and-api-updates/?utm\_source=chatgpt.com

14. Press, O., Wolf, L.: Using the output embedding to improve language models. In: Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers. pp. 157–163 (2017). https: //doi.org/10.48550/arXiv.1608.05859

15. Seznam.cz: Diana hlaváčová: Sellma aneb jak v seznamu krotíme dravé jazykové modely? https://blog.seznam.cz/2024/10/diana-hlavacova-sellma-aneb-j ak-v-seznamu-krotime-drave-jazykove-modely/ (2024), accessed: 2026-02-09

16. Seznam.cz: Peter pekarovič and martin kirschner: Seznam ai. technologie, která není jen chytrá, ale hlavně užitečná. https://blog.seznam.cz/2025/10/peter-p ekarovic-martin-kirschner-seznam-ai-technologie-ktera-neni-jen-chytr a-ale-hlavne-uzitecna/ (2025), accessed: 2026-02-09

17. Sturua, S., Mohr, I., Akram, M.K., Günther, M., Wang, B., Krimmel, M., Wang, F., Mastrapas, G., Koukounas, A., Wang, N., et al.: jina-embeddings-v3: Multilingual embeddings with task lora. arXiv preprint arXiv:2409.10173 (2024). https://do i.org/10.48550/arXiv.2409.10173

18. Wang, Q., Fang, Y., Ravula, A., Feng, F., Quan, X., Liu, D.: Webformer: The web-page transformer for structure information extraction. In: Proceedings of the ACM Web Conference 2022. pp. 3124–3133 (2022). https://doi.org/10.1145/34 85447.3512032

19. Warner, B., Chafin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., et al.: Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory eficient, and long context finetuning and inference. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 2526–2547 (2025). https://doi.org/10.18653/v1/2025.acl-long.127

20. Wijmans, E., Huval, B., Hertzberg, A., Koltun, V., Krähenbühl, P.: Cut your losses in large-vocabulary language models. arXiv preprint arXiv:2411.09009 (2024). ht tps://doi.org/10.48550/arXiv.2411.09009

21. Wu, Y., Schuster, M., Chen, Z., Le, Q.V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., et al.: Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144 (2016). https://doi.org/10.48550/arXiv.1609.08144

22. Xu, J., Cao, Y., Li, H., Craswell, N., Huang, Y.: Searching documents based on relevance and type. In: European Conference on Information Retrieval. pp. 629– 636. Springer (2007). https://doi.org/10.1007/978-3-540-71496-5\_60

23. Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., et al.: Qwen3 embedding: Advancing text embedding and reranking

through foundation models. arXiv preprint arXiv:2506.05176 (2025). https://do i.org/10.48550/arXiv.2506.05176

24. Zhang, Z., Yu, B., Liu, T., Liu, T., Wang, Y., Guo, L.: Learning structural co-occurrences for structured web data extraction in low-resource settings. In: Proceedings of the acm web conference 2023. pp. 1683–1692 (2023). https: //doi.org/10.1145/3543507.3583387

25. Zhuang, H., Emma Zhang, W., Yang, J., Chen, W., Sheng, Q.Z.: Not all negatives are equally negative: Soft contrastive learning for unsupervised sentence representations. In: Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. pp. 3591–3601 (2024). https://doi.org/10.1145/ 3627673.3679745