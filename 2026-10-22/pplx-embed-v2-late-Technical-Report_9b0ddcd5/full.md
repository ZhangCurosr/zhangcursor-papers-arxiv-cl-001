# pplx-embed-v2-late Technical Report

Markus Krimmel<sup>\*,1</sup>, Louis Milliken<sup>\*,1</sup>, Ekaterina Glazkova<sup>1</sup>, Maximilian Schall<sup>1</sup>, Antoine Chafin<sup>1</sup>, Sedigheh Eslami<sup>1</sup>, Jonas Golde<sup>1</sup>, Denis Bykov<sup>1</sup>, Claire Beamer<sup>1</sup>, Jeremy Yang<sup>1</sup> and Bo Wang<sup>1</sup>   
<sup>\*</sup>Equal contributions, <sup>1</sup>Perplexity AI

## Abstract

In this report, we introduce pplx-embed-v2-late, a pair of 0.6B and 9B multilingual and multimodal late-interaction models. Both models achieve state-of-the-art results within their respective size classes across a range of text, image, and visual document retrieval benchmarks. On visual document retrieval tasks, pplx-embed-v2-late-0.6b is competitive with models several times larger than itself. The models are trained via representation distillation from an 18B teacher. Consequently, they share an embedding space that supports asymmetric deployment: documents can be indexed with the 9B model and queried with the low-latency 0.6B model. Both models are available at https://huggingface.co/collections/perplexity-ai/pplx-embed-v2.

## 1. Introduction

Retrieval systems face a trade-of between eficiency and the depth of interaction between queries and documents. Dense embeddings interact via cheap inner products, which allow for eficient approximate nearest neighbor search. At the other extreme, cross-encoders process the query and document together, enabling richer interaction but requiring a separate forward pass for every ranked candidate. Late-interaction models (Khattab and Zaharia, 2020) occupy the space between the two. Documents are indexed ofline and represented as bags of embedding vectors. These document vectors interact with query vectors via the MaxSim rule, allowing richer interaction than dense embeddings (Jayaram, 2026). Late-interaction models have shown especially strong performance in out-of-domain settings (Clavié, 2024; Santhanam et al., 2022), long-context retrieval (Chafin, 2025a; Takehi et al., 2025; Warner et al., 2025), agentic search (Chafin, 2026; Mixedbread Team, 2026), and visual document retrieval (VDR) (Faysse et al., 2025; Günther et al., 2025).

In this report, we introduce pplx-embed-v2-late, a pair of 0.6B and 9B multimodal and multilingual late-interaction models. Both models are distilled at the representation level from a contrastively trained 18B teacher. We find that representation distillation outperforms contrastive training alone, especially at smaller model scales. Since the 0.6B and 9B models are trained to match the same teacher, the two encoders are interchangeable: a query embedded by the 0.6B model can be scored directly against an index built by the 9B model, and vice versa. This allows for hybrid deployment options where the latency-critical query path can be served by the fast 0.6B model while documents are indexed ofline via the stronger 9B model, with the cost of indexing amortized over the lifetime of the index. Both models individually achieve state-of-the-art performance on a range of text, image, and VDR benchmarks, including ViDoRe-V3 (Loison et al., 2026). In concurrent work, Liu et al. (2026) study representation-level distillation in late-interaction models to produce small query encoders. Our work distills query and document token representations into compact multimodal late-interaction models that can serve as both query and document encoders.

## 2. Training

In the following, we describe the training procedure for the 18B teacher model and the 9B and 0.6B students. The teacher model is trained contrastively via an InfoNCE (van den Oord et al., 2018) objective in two stages: first on query-document pairs, then on triplets of queries, relevant documents, and hard negatives. The students are trained via a pointwise distillation objective similar to that of LEAF (Vujanic and Rückstieß, 2026) to match the embeddings of the teacher. We reuse the pair data from the contrastive training stage for distillation. All of our late-interaction models have 128-dimensional embeddings.

## 2.1. Contrastive Teacher Training

We use Qwen3.5 (Qwen Team, 2026) as the backbone for our late-interaction models. Qwen3.5 uses a hybrid architecture consisting of gated full attention layers with a causal mask and recurrent Gated DeltaNet (Yang et al., 2025) (GDN) layers. We disable the causal mask in the full attention layers, yielding a hybrid bidirectional-causal architecture. We initialize the 18B late-interaction model by pruning the last 20 layers from Qwen3.5-27B. This reduces the cost of distillation while maintaining a strong training signal.

Pair Training. Our pair data is composed of 28 datasets which we further partition by language and domain to increase the dificulty of in-batch negatives. This yields 981 sub-datasets with 186M rows of query-document pairs in total, 88.3% of which are texttext pairs, 8.3% are text-image pairs and 3.4% are text-visual document pairs. Of the text-text pairs, 14M are synthesized by generating queries for specific passages in PDF documents from the FinePDFs (Kydlíček et al., 2025) dataset. During training, we adjust the sampling weights of each dataset to reach a data mixture of 56.5% text-text pairs 30.9% text-image pairs, and 12.6% text-visual document pairs. Following previous work (Günther et al., 2023), we sample full batches from the sub-datasets. Since many sub-datasets are domain or language slices of the same source dataset, we introduce a novel quasi-random sampling strategy to ensure that sub-datasets are sampled both (i) in proportions close to the intended mixture and (ii) at regular intervals across training steps, avoiding both long gaps and temporal clustering. The sampling should satisfy these properties not only at the level of sub-datasets but also at the source-dataset level. We achieve this by constructing a cumulative sampling distribution in which all sub-datasets from the same source occupy contiguous intervals and mapping a randomly shifted golden-ratio Kronecker sequence through its inverse CDF. The sequence’s low-discrepancy coverage (Kuipers and Niederreiter, 1974, Chapter 2, Theorem 3.4) and structured return times (Slater, 1950) apply at both levels, guaranteeing proportional, well-spaced sampling of individual slices and their parent sources. We illustrate this in Figure 1.

We train the model with LoRA (Hu et al., 2022) on all attention and feed-forward projections to stabilize training and mitigate catastrophic forgetting. The adapters have rank 128. Our training objective is the InfoNCE loss with a contrastive batch size of 16,384 query-document pairs. Following (Zhang et al., 2025), we mask in-batch negatives based on a margin of 0.2 to the query-positive MaxSim score. During training, we truncate queries to 256 tokens and documents to 1,000 tokens, and resize images to at most 917,504 pixels while preserving their original aspect ratios. To enable the large contrastive batch size, we use gradient caching (Gao et al., 2021) and implement the MaxSim operation as a custom Triton (Tillet et al., 2019) kernel to avoid materializing token-level similarity matrices. This work was concurrent with the eficient MaxSim implementations by Lac and Wu (2026); Pony et al. (2026); Sharma (2026).

![](images/23a8f249f6e4dc47d41ed5c4de7dbf5cf2a0e60c5d5cb02056bfc74cff702f7f.jpg)  
Figure 1 | Comparison of our sampling method with independent random sampling according to the dataset weights.

Triplet Training. After pair training, we merge the LoRA weights into the backbone and add a new adapter with rank 32. We freeze the vision tower and train only on text triplets. We use at most four hard negatives per query and contrast each positive against its corresponding hard negatives and all other in-batch documents. Our triplet data is composed of public triplet datasets, mined hard negatives, and synthetic triplets created by modifying key information in highly templated documents.

## 2.2. Representation Distillation

To distill our 18B teacher model down to 0.6B and 9B student models, we train the students to reproduce the teacher’s token-level representations by minimizing the Euclidean distance between corresponding ℓ -normalized embeddings. We average these distances over unmasked query and document positions separately, then combine the two averages with equal weight. The teacher remains frozen, so gradients update only the student. We let � and � denote the unmasked query-token and document-token positions across the batch and indicate student and teacher embeddings by superscripts � and �. With $\mathbf { q } _ { i t }$ and $\mathbf { d } _ { i t }$ denoting query and document embeddings at token position � of example � within the batch, we formulate the loss as:

$$
\mathcal { L } = \frac { 1 } { \vert \boldsymbol { \mathcal { Q } } \vert } \sum _ { ( i , t ) \in \mathcal { Q } } \left. \mathbf { q } _ { i t } ^ { \mathrm { S } } - \mathbf { q } _ { i t } ^ { \mathrm { T } } \right. _ { 2 } + \frac { 1 } { \vert \boldsymbol { \mathcal { D } } \vert } \sum _ { ( i , t ) \in \mathcal { D } } \left. \mathbf { d } _ { i t } ^ { \mathrm { S } } - \mathbf { d } _ { i t } ^ { \mathrm { T } } \right. _ { 2 } .
$$

Since this loss is not contrastive, we can mix sub-datasets when sampling a batch to reduce gradient noise without weakening our training signal.

Distillation to 9B. We use Qwen3.5-9B with bidirectional full attention as the backbone for our 9B late-interaction model. Unlike during contrastive training of the 18B model,

LoRA does not appear to provide suficient capacity to achieve satisfactory downstream performance with representation distillation. We hypothesize that this is caused by the constraints imposed on the embedding space by the distillation objective. To increase capacity while limiting catastrophic forgetting, we fully fine-tune the last eight layers of the text tower while applying LoRA to the remaining projections.

Distillation to 0.6B. The 0.6B late-interaction model is initialized from a pruned version of Qwen3.5-0.8B with bidirectional full attention layers. Within each block of Qwen3.5-0.8B, we prune two of the three GDN layers, producing an architecture that alternates between gated full attention and GDN layers. The resulting model has 594M parameters in total, of which 100M are used for image encoding and 254M are in the token-embedding table. Hence, only about 240M parameters are active for text encoding and 340M for image encoding. This small size allows deployment on edge devices. We find that, especially on VDR tasks, full fine-tuning of the 0.6B backbone outperforms parameter-eficient approaches. Hence, in contrast to the 18B and 9B models, we fine-tune all parameters.

## 3. Evaluation

We evaluate on various text-only and text-to-image benchmarks. We present results on single-hop and agentic retrieval tasks. We excluded any training data that overlaps with evaluated tasks.

## 3.1. Text retrieval

For text retrieval, we evaluate a collection of 72 public tasks across six domains: finance (18 tasks), legal (20), health (8), conversation (5), tech (4), and multilingual (17), which we bundle into an internal domain-specific benchmark. Additionally, we evaluate the text version of ViDoRe-V3 (Loison et al., 2026) and the Q2D-Web (Schall et al., 2026) benchmark. Text baselines include Nemotron 3 Embed 8B (NVIDIA, 2026), Qwen3 Embedding (Zhang et al., 2025), Gemini Embedding 2 (Shanbhogue et al., 2026), and Voyage 4 (Voyage AI, 2026).

Domain-Specific Benchmark. The domain-specific collection is built from public datasets such as FinQA (Chen et al., 2021), AILACasedocs (Bhattacharya et al., 2020) and XLSum (Hasan et al., 2021). We do not disclose the full task list to reduce the risk of future models being tuned to it; as a result, these scores cannot be independently reproduced. pplx-embed-v2-late-9b averages 81.3% nDCG@10, 1.6 pp above the next best model, and scores highest on every domain except health, where nemotron-3-embed-8b and voyage-4-large score higher. pplx-embed-v2-late-0.6b averages 78.0% and scores highest among models with fewer than 1B parameters on every domain (Table 1).

Q2D-Web. The Q2D-Web benchmark contains about 70,000 agent-reformulated queries from production search trafic over a corpus of 190M web documents. Because it targets first-stage retrieval, we report Recall@1000. Relevance is judged in three ways: documents cited by the agent (Citation), documents ranked highly by our production retrieval system (Web), and the union of both extended with LLM judgments (Combined). We focus on Combined, which reduces bias towards models that rank similarly to the production system. On Combined, the 9B and 0.6B models score 74.8% and 73.6%, compared with 69.3% for the strongest baseline (Table 2).

Table 1 | Domain-specific text retrieval performance (nDCG@10, %). The average weights the six domains equally; parentheses show the number of tasks in each domain. Dim. is the embedding dimension (per token for multi-vector models). Best score per column in bold.
<table><tr><td>Model</td><td>| Dim. Vectors ∥| Avg.</td><td></td><td></td><td>Finance (18)</td><td>Legal (20)</td><td>Health (8)</td><td>Conv. (5)</td><td>Tech (4)</td><td>Multiling. (17)</td></tr><tr><td colspan="10">&gt;1B / size unreported</td></tr><tr><td> $\mathtt { p p 1 x - e m b e d - v 2 - l a t e - 9 b }$ </td><td>128</td><td>multi</td><td>81.3</td><td>82.5</td><td>81.6</td><td>82.5</td><td>81.5</td><td>70.4</td><td>89.5</td></tr><tr><td>nemotron-3-embed-8b</td><td>4096</td><td>single</td><td>79.7</td><td>80.8</td><td>81.1</td><td>83.9</td><td>77.4</td><td>67.5</td><td>87.9</td></tr><tr><td> $\mathtt { g e m i n i - e m b e d d i n g - 2 }$ </td><td>3072</td><td>single</td><td>78.3</td><td>77.0</td><td>78.5</td><td>82.4</td><td>76.2</td><td>67.7</td><td>88.1</td></tr><tr><td>voyage-4-large</td><td>1024</td><td>single</td><td>79.0</td><td>81.5</td><td>80.0</td><td>83.0</td><td>76.7</td><td>65.1</td><td>87.9</td></tr><tr><td>qwen3-embedding-8b</td><td>4096</td><td>single</td><td>74.8</td><td>72.3</td><td>76.8</td><td>80.2</td><td>72.0</td><td>62.7</td><td>84.7</td></tr><tr><td colspan="10"> $< 1 B$ </td></tr><tr><td> $\mathtt { p p 1 x - e m b e d - v 2 - l a t e - 0 . 6 b }$ </td><td>128</td><td>multi</td><td>78.0</td><td>79.9</td><td>78.2</td><td>78.0</td><td>79.7</td><td>66.6</td><td>85.8</td></tr><tr><td>voyage-4-nano</td><td>2048</td><td>single</td><td>74.6</td><td>77.3</td><td>75.8</td><td>76.0</td><td>74.7</td><td>59.4</td><td>84.3</td></tr><tr><td>qwen3-embedding-0.6b</td><td>1024</td><td>single</td><td>68.4</td><td>67.9</td><td>70.0</td><td>72.4</td><td>68.4</td><td>53.9</td><td>77.8</td></tr></table>

Table 2 | Text retrieval on the public subset of ViDoRe-V3 after converting page images to Markdown with OCR (average nDCG@10, %) and on Q2D-Web (Recall@1000, %) under three relevance judgment sets. Dim. is the embedding dimension (per token for multi-vector models). topk-embed-v1 ViDoRe-V3 scores are taken from the model cards (TopK, 2026a,b) at 128 dimensions; on Q2D-Web, these models were evaluated with an output dimension of 128. – denotes not evaluated. Best score per column in bold.
<table><tr><td>Model</td><td>Dim. Vectors</td><td></td><td>ViDoRe-V3 (text)</td><td colspan="3">Q2D-Web Citation Web Combined</td></tr><tr><td></td><td></td><td></td><td>Avg.</td><td></td><td></td><td></td></tr><tr><td colspan="7">&gt;1B / size unreported</td></tr><tr><td>pplx-embed-v2-late-9b</td><td>128</td><td>multi</td><td>64.7</td><td>88.2</td><td>68.8</td><td>74.8</td></tr><tr><td>nemotron-3-embed-8b</td><td>4096</td><td>single</td><td>60.6</td><td>83.1</td><td>61.7</td><td>69.3</td></tr><tr><td>topk-embed-v1-2b</td><td>128</td><td>multi</td><td>59.2</td><td>79.2</td><td>57.7</td><td>65.3</td></tr><tr><td>gemini-embedding-2</td><td>3072</td><td>single</td><td>58.9</td><td></td><td></td><td></td></tr><tr><td>voyage-4-large</td><td>1024</td><td>single</td><td>58.9</td><td></td><td></td><td></td></tr><tr><td>qwen3-embedding-8b</td><td>4096</td><td>single</td><td>51.5</td><td>79.3</td><td>58.3</td><td>65.3</td></tr><tr><td colspan="7">&lt;1B</td></tr><tr><td>pplx-embed-v2-late-0.6b</td><td>128</td><td>multi</td><td>61.2</td><td>86.7</td><td>67.3</td><td>73.6</td></tr><tr><td>topk-embed-v1-0.8b</td><td>128</td><td>multi</td><td>56.9</td><td>77.4</td><td>56.2</td><td>63.6</td></tr><tr><td>voyage-4-nano</td><td>2048</td><td>single</td><td>53.3</td><td>69.5</td><td>50.1</td><td>57.4</td></tr><tr><td>qwen3-embedding-0.6b</td><td>1024</td><td>single</td><td>44.2</td><td>71.2</td><td>51.8</td><td>58.7</td></tr></table>

ViDoRe-V3. On the Markdown version of ViDoRe-V3 (Table 2), both pplx-embed-v2- late models outperform all evaluated baselines, with the 9B model more than 4 pp above the best competitor.

## 3.2. Text-to-image retrieval

ViDoRe-V3. Our primary VDR benchmark is the image version of ViDoRe-V3. We compare against dense (qwen3-vl-embed-8b (Li et al., 2026), gemini-embedding-2) and lateinteraction (topk-embed-v1 0.8B and 2B) vision-language models, and against two visiononly late-interaction model families trained specifically for image retrieval (EVIE (Wang et al., 2026) and Nemotron ColEmbed V2 (de Souza P. Moreira et al., 2026)). Among vision-language models, pplx-embed-v2-late-9b and pplx-embed-v2-late-0.6b have the two highest averages. pplx-embed-v2-late-9b scores 65.2%, below EVIE-8B (66.4%) and EVIE-4.5B (65.9%), which have output dimensions of 4,096 and 2,048 per token, respectively, compared to pplx-embed-v2-late’s output dimension of 128. pplx-embedv2-late-0.6b scores 62.3%, 0.9 pp below topk-embed-v1-2b at its full width of 2,048 dimensions per token and 0.4 pp above it at 128 dimensions, 3.3 pp higher than topkembed-v1-0.8b at 128 dimensions.

Table 3 | Text-to-image retrieval (nDCG@10, %) on the public subset of ViDoRe-V3 (average), MIRACL-Vision and PPLX-Q2I. Vision-only models are trained specifically for image retrieval and are listed separately from vision-language models. Dim. is the embedding dimension (per token for multi-vector models). Nemotron ColEmbed and EVIE models were not evaluated on PPLX-Q2I because their high output dimensionality makes indexing impractical; – denotes not evaluated. Best score per column in bold.
<table><tr><td>Model</td><td>Dim. Vectors</td><td></td><td>ViDoRe-V3 (image)</td><td>MIRACL-Vision</td><td>PPLX-Q2I</td></tr><tr><td colspan="6">Vision + language, &gt;1B / size unreported</td></tr><tr><td>pplx-embed-v2-late-9b</td><td>128</td><td>multi</td><td>65.2</td><td>75.7</td><td>65.0</td></tr><tr><td>topk-embed-v1-2b</td><td>128</td><td>multi</td><td>61.9</td><td>60.2</td><td>51.5</td></tr><tr><td>qwen3-vl-embedding-8b</td><td>4096</td><td>single</td><td>58.3</td><td>68.5</td><td>50.6</td></tr><tr><td>gemini-embedding-2</td><td>3072</td><td>single</td><td>46.0</td><td>77.5</td><td>67.0</td></tr><tr><td colspan="6">Vision + language, &lt;1B</td></tr><tr><td>pplx-embed-v2-late-0.6b</td><td>128</td><td>multi</td><td>62.3</td><td>67.5</td><td>60.4</td></tr><tr><td>topk-embed-v1-0.8b</td><td>128</td><td>multi</td><td>59.0</td><td>56.5</td><td>42.3</td></tr><tr><td colspan="6">Vision-only</td></tr><tr><td>nemotron-colembed-v2-8b</td><td>4096</td><td>multi</td><td>63.5</td><td>68.6</td><td></td></tr><tr><td>nemotron-colembed-v2-4b</td><td>2560</td><td>multi</td><td>61.4</td><td>62.7</td><td></td></tr><tr><td>evie-8b</td><td>4096</td><td>multi</td><td>66.4</td><td></td><td></td></tr><tr><td>evie-4.5b</td><td>2048</td><td>multi</td><td>65.9</td><td></td><td></td></tr></table>

MIRACL-Vision. The MIRACL-Vision benchmark (Osmulski et al., 2025) consists of a corpus of Wikipedia article screenshots. On this task, the vision-only Nemotron ColEmbed V2 4B and 8B models score 62.7% and 68.6%, respectively; pplx-embed-v2-late-0.6b scores 67.5% and pplx-embed-v2-late-9b scores 75.7%, 1.8 percentage points below the highest-scoring model, gemini-embedding-2 (Table 3).

PPLX-Q2I. PPLX-Q2I is built from production image-search logs, with 10,000 queries and 100,000 images (both visual documents and natural images); a query’s relevant images are those served for it in production. The 0.6B and 9B pplx-embed-v2-late models score 60.4% and 65.0%, compared with 67.0% for gemini-embedding-2 and 50.6% for qwen3-vl-embedding-8b (Table 3).

![](images/14f942cae0f46533285e11812fc2a633b17ca4a7d70920944842d85b6d59b1f3.jpg)  
Figure 2 | Cross-model retrieval nDCG@10 for three query/document model pairings. Results across six text domains.

## 3.3. Cross-model retrieval

Because the two models share an embedding space, a corpus can be indexed with the 9B model and queried with the 0.6B model, keeping query encoding cost the same as in the 0.6B→0.6B setting. We illustrate this in Figure 2 on the domain-specific benchmarks. The 0.6B→9B setting scores higher than 0.6B→0.6B on every domain, by 1.6 pp on the domain average. The 9B→9B setting scores highest at 81.3%. The hybrid setting thus improves retrieval performance without increasing query encoding cost.

## 3.4. Agentic retrieval

In BrowseComp+ (Chen et al., 2026), an LLM agent (GPT-OSS-120B (OpenAI, 2025), high efort) answers complex queries by searching a corpus with the embedding model under test, and a second LLM judges the answers. Both pplx-embed-v2-late models achieve higher accuracy with fewer searches than all evaluated baselines (Figure 3). The 9B model reaches 64.0% accuracy, 4.9 pp above Reason-ModernColBERT (Chafin, 2025b), the only other late-interaction model evaluated, and 8.7 pp above the best dense model, voyage-4-large.

![](images/2eac9e173ab014091b1117605d6987508626e041848096362e10d452ca4ec3ab.jpg)

![](images/833ac4e83ab658b0ea78feb9f1a5da2898d626dfa799c1fc0c14233fd172aec1.jpg)  
Figure 3 | BrowseComp+ accuracy and recall versus the average number of searches (GPT-OSS-120B agent, high efort).

## 4. Conclusion

We presented pplx-embed-v2-late, a pair of 0.6B and 9B multimodal, multilingual lateinteraction models distilled at the representation level from a contrastively trained 18B teacher. On our domain-specific text benchmarks, Q2D-Web and the Markdown version of ViDoRe-V3, both models outperform all evaluated baselines compared, to other models of their size. On ViDoRe-V3 image retrieval, they have the highest averages among visionlanguage models, while vision-only EVIE models score higher. Because both students are trained to match the same teacher, they share an embedding space: indexing with the 9B model and querying with the 0.6B model improves over using the 0.6B model alone, at the same query encoding cost. In agentic retrieval, both models reach higher BrowseComp+ accuracy with fewer searches than all evaluated baselines.

## References

P. Bhattacharya, K. Ghosh, S. Ghosh, A. Pal, P. Mehta, A. Bhattacharya, and P. Majumder. AILA 2019 precedent & statute retrieval task. Zenodo dataset, Oct. 2020.

A. Chafin. GTE-ModernColBERT, 2025a. URL https://huggingface.co/lightonai/ GTE-ModernColBERT-v1.

A. Chafin. Reason-ModernColBERT, 2025b. URL https://huggingface.co/lightonai/ Reason-ModernColBERT.

A. Chafin. Agent-ModernColBERT, 2026. URL https://huggingface.co/lightonai/ Agent-ModernColBERT.

Z. Chen, W. Chen, C. Smiley, S. Shah, I. Borova, D. Langdon, R. Moussa, M. Beane, T.-H. Huang, B. Routledge, and W. Y. Wang. FinQA: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.emnlp-main.300.

Z. Chen, X. Ma, S. Zhuang, P. Nie, K. Zou, S. Sharifymoghaddam, A. Liu, J. Green, K. Patel, R. Meng, M. Su, Y. Li, H. Hong, X. Shi, X. Liu, H. Oyarhoseini, N. Thakur, C. Zhang, L. Gao, W. Chen, and J. Lin. BrowseComp-Plus: A fair and disentangled evaluation benchmark for deep search agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 22349–22370. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long. 1023.

B. Clavié. JaColBERTv2.5: Optimising multi-vector retrievers to create state-of-the-art japanese retrievers with constrained resources. arXiv preprint arXiv:2407.20750, 2024.

G. de Souza P. Moreira, R. Ak, M. Xu, O. Holworthy, B. Schiferer, Z. Yu, Y. Babakhin, R. Osmulski, J. Cai, R. Chesler, B. Liu, and E. Oldridge. Nemotron ColEmbed V2: Top-performing late interaction embedding models for visual document retrieval. In Proceedings of the 1st Late Interaction Workshop (LIR) @ ECIR 2026, 2026.

M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, and P. Colombo. ColPali: Eficient document retrieval with vision language models. In International Conference on Learning Representations (ICLR), 2025.

L. Gao, Y. Zhang, J. Han, and J. Callan. Scaling deep contrastive learning batch size under memory limited setup. In Proceedings of the 6th Workshop on Representation Learning for NLP, pages 316–321. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.repl4nlp-1.31.

M. Günther, L. Milliken, J. Geuter, G. Mastrapas, B. Wang, and H. Xiao. Jina embeddings: A novel set of high-performance sentence embedding models. In Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023), pages 8–18. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023. nlposs-1.2.

M. Günther, S. Sturua, M. K. Akram, I. Mohr, A. Ungureanu, B. Wang, S. Eslami, S. Martens, M. Werk, N. Wang, and H. Xiao. jina-embeddings-v4: Universal embeddings for multimodal multilingual retrieval, 2025. URL https://arxiv.org/abs/2506.18902.

T. Hasan, A. Bhattacharjee, M. S. Islam, K. Mubasshir, Y.-F. Li, Y.-B. Kang, M. S. Rahman, and R. Shahriyar. XL-Sum: Large-scale multilingual abstractive summarization for 44 languages. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4693–4703. Association for Computational Linguistics, 2021. doi: 10.18653/ v1/2021.findings-acl.413.

E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

R. Jayaram. Multi-vector embeddings are provably more expressive than single vector embeddings. arXiv preprint arXiv:2606.23475, 2026.

O. Khattab and M. Zaharia. ColBERT: Eficient and efective passage search via contextualized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, pages 39–48. ACM, July 2020. doi: 10.1145/3397271.3401075.

L. Kuipers and H. Niederreiter. Uniform Distribution of Sequences. John Wiley & Sons, 1974.

H. Kydlíček, G. Penedo, and L. von Werra. FinePDFs: Liberating 3t of the finest tokens from PDFs. https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog, 2025. Blog post.

A. Lac and T. Wu. late-interaction-kernels: Fused Triton kernels for late-interaction scoring. Software, 2026. URL https://github.com/hcompai/late-interaction-kernels.

M. Li, Y. Zhang, D. Long, K. Chen, S. Song, S. Bai, Z. Yang, P. Xie, A. Yang, D. Liu, J. Zhou, and J. Lin. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

Z. Liu, Z. Wang, Y. Zhang, and Y. Xiao. ColNanoVDR: Document-free query distillation for multi-vector visual document retrieval via optimal transport. arXiv preprint arXiv:2609.34899, 2026.

A. Loison, Q. Macé, A. Edy, V. Xing, T. Balough, G. d. S. P. Moreira, B. Liu, M. Faysse, C. Hudelot, and G. Viaud. ViDoRe V3: A comprehensive evaluation of retrieval augmented generation in complex real-world scenarios. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16570–16600. Association for Computational Linguistics, 2026. doi: 10.18653/v1/ 2026.acl-long.755.

Mixedbread Team. Beyond the limit: Introduce Mixedbread Wholembed v3. Mixedbread Blog, Mar. 2026. URL https://www.mixedbread.com/blog/wholembed-v3.

NVIDIA. Nemotron-3-Embed-8B-BF16. Model card, July 2026. URL https:// huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16.

OpenAI. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

R. Osmulski, G. de Souza P. Moreira, R. Ak, M. Xu, B. Schiferer, and E. Oldridge. MIRACL-VISION: A large, multilingual, visual document retrieval benchmark. arXiv preprint arXiv:2505.11651, 2025.

R. Pony, D. Ezer, A. Raz Goldfarb, I. Friedman, O. Naparstek, and U. Barzelay. Flash-MaxSim: IO-aware fused kernels for late-interaction retrieval. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing: Industry Track, 2026. To appear.

Qwen Team. Qwen3.5: Towards native multimodal agents, Feb. 2026. URL https: //qwen.ai/blog?id=qwen3.5.

K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia. ColBERTv2: Efective and eficient retrieval via lightweight late interaction. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), pages 3715–3734. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.naacl-main.272.

M. Schall, S. Eslami, M. Krimmel, A. Chafin, L. Milliken, B. Wang, and D. Bykov. Q2D-Web: A large-scale benchmark for retrieval in agentic RAG systems. arXiv preprint arXiv:2609.08887, 2026.

M. Shanbhogue, Z. Li, S. Zhang, G. H. Ábrego, S.-C. Huang, A. Jain, D. Salz, S. Goenka, C. Hegde, J. Ma, F. Chen, J. Wu, T. Dabral, B. Samari, K. Poulet, D. Cer, K. Chen, P. Suganathan, H. Hui, J. Andonov, P. Schlattner, J. Han, I. Naim, W. Lowe, V. Pchelin, A. Yang, Y.-T. Chen, Z. Ding, G. Zhang, G. Heigold, Y. Chen, A. Reveillon, B. Mccloskey, W. Zhou, D. Kim, R. Meng, E. Wang, J. Zheng, H. Fede, Z. Yang, K. Mosley, B. Potetz, S. Dua, H. S. Vera, S. Gao, H. Zhang, A. Hess, H. Ying, A. Montes, K. Gill, M. Choi, S. Russo, A. Hauth, J. Lee, M. Boratko, M. Barnes, V. Rao, C. Musat, C. Allauzen, E. Variani, S. Kumar, T. Bagby, J. Jiao, Y. Gu, T. Li, A. Agrawal, R. Santana, D. Nath, S. Karukas, S. Han, L. Loher, A. Twu, N. Vyas, S. Bhai, F. P. Gomez, W. Zhang, C. Liu, J. Yang, S. Qiu, S. Zhang, S. Kulkarni, S. Rothe, S. Nakamoto, R. Hofmann, Z. Gleicher,

Y. Sung, Q. Yin, T. Duerig, and M. Seyedhosseini. Gemini Embedding 2: A native multimodal embedding model from Gemini. arXiv preprint arXiv:2605.27295, 2026.

A. Sharma. TileMaxSim: IO-aware GPU MaxSim scoring with dimension tiling and fused product quantization. arXiv preprint arXiv:2606.26439, 2026.

N. B. Slater. The distribution of the integers � for which {��} < �. Mathematical Proceedings of the Cambridge Philosophical Society, 46(4):525–534, 1950. doi: 10.1017/ S0305004100026086.

R. Takehi, B. Clavié, S. Lee, and A. Shakir. Fantastic (small) retrievers and how to train them: mxbai-edge-colbert-v0 tech report. arXiv preprint arXiv:2510.14880, 2025.

P. Tillet, H. T. Kung, and D. Cox. Triton: An intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19. ACM, 2019. doi: 10.1145/3315508.3329973.

TopK. topk-embed-v1-small. Model card, 2026a. URL https://huggingface.co/ topk-io/topk-embed-v1-small.

TopK. topk-embed-v1-xsmall. Model card, 2026b. URL https://huggingface.co/ topk-io/topk-embed-v1-xsmall.

A. van den Oord, Y. Li, and O. Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Voyage AI. The Voyage 4 model family: shared embedding space with MoE architecture. Voyage AI Blog, Jan. 2026. URL https://blog.voyageai.com/2026/01/15/ voyage-4/.

R. Vujanic and T. Rückstieß. LEAF: Knowledge distillation of text embedding models with teacher-aligned representations. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 43362–43383. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long. 2008.

Z. Wang, W. Wen, Q. Ji, Q.-W. Zhang, R. Qiao, and X. Sun. EVIE: Evidence-vectorinformed embeddings for visual document retrieval. arXiv preprint arXiv:2610.11553, 2026.

B. Warner, A. Chafin, B. Clavié, O. Weller, O. Hallström, S. Taghadouini, A. Gallagher, R. Biswas, F. Ladhak, T. Aarsen, N. Cooper, G. T. Adams, J. Howard, and I. Poli. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory eficient, and long context finetuning and inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2526–2547. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025. acl-long.127.

S. Yang, J. Kautz, and A. Hatamizadeh. Gated delta networks: Improving Mamba2 with delta rule. In International Conference on Learning Representations (ICLR), 2025.

Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, et al. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.