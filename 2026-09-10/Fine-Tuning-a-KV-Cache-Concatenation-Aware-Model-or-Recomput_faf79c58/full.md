# Fine-Tuning a KV Cache Concatenation-Aware Model or Recomputing KV Caches? Why Not Both?

Fumihiko Tachibana

Daisuke Miyashita

Jun Deguchi

Kioxia Corporation

## Abstract

In Retrieval-Augmented Generation (RAG) systems, a large number of retrieved chunks are concatenated to form the input context so that users can receive high-quality responses based on external knowledge. As a result, the input context length increases substantially, leading to a larger prefill workload and, in turn, a longer time to first token (TTFT). While previous works that reuse precomputed key-value (KV) caches effectively reduce TTFT for longcontext inputs, it remains unclear whether response quality is preserved when the input context becomes very long. In this paper, we propose a combined approach that (i) fine-tunes the model while taking KV cache concatenation into account and (ii) selectively recom putes a subset of the KV caches. By applying both techniques, we demonstrate improved accuracy for long-context inputs. Experiments on the RULER benchmark show that, for a 124ktoken input, our method improves the RULER score by 9.7 point over the baseline that recomputes KV caches only. Moreover, TTFT is reduced by 80% compared with full attention.

## 1 Introduction

When a large language model (LLM) receives additional chunks retrieved by Retrieval-Augmented Generation (RAG) (Li et al., 2022; Gao et al., 2024), the total input context length increases substantially. Since prefill computation scales quadratically with input context length, computing the keyvalue (KV) caches incurs a large computational cost, which in turn increases the time to first token (TTFT).

One approach for reducing TTFT is to reuse precomputed KV caches for the context that is frequently reused. First, prefix caching stores and reuses KV caches when the context begins with the same initial token sequence (Zheng et al., 2024). Prefix caching does not degrade generation quality because the KV caches for the prefix context are independent of the subsequent text. However, this reuse is only valid when the input tokens exactly match those of the stored KV caches. This constraint limits the reusability of KV caches.

Second, KV cache concatenation aims to reuse KV caches even when the context corresponding to the precomputed KV caches is not a prefix, thereby increasing their reusability. However, because precomputed KV caches are generated without preceding chunks, reused KV caches for non-prefix tokens do not include cross-attention from preceding chunks, which introduces deviations in the KV caches compared with those computed with full attention. When such a mismatch occurs, the attention outputs change, and generation quality degrades even if the positional embeddings of the KV caches are adjusted to match the current context positions (He et al., 2025).

To mitigate this problem, several methods have been proposed. CacheBlend (Yao et al., 2025) and EPIC (Hu et al., 2025) selectively recompute portions of the KV caches based on certain criteria. Other approaches, such as Block-attention (Ma et al., 2025), TurboRAG (Lu et al., 2025), and KVLink (Yang et al., 2025d), fine-tune the model under the assumption that there is no crossattention between chunks, allowing multiple KV caches to be concatenated and reused. Although prior techniques improve generation quality on several benchmarks, their response accuracy under long input contexts remains underexplored. We show that accuracy consistently declines as context length increases, revealing that existing methods are inadequate for long-context settings. Since the benefit of KV cache reuse grows with context length, developing methods that preserve accuracy for long inputs is essential.

In this paper, we demonstrate that combining model fine-tuning with selective recomputation of a small fraction of the KV caches enables more robust handling of long-context inputs. We demonstrate that using both methods suppresses the deviation in the key entries (i.e., the values of the key vectors) more effectively than using either method alone because the two approaches contribute differently to reduce these deviations. As a result, our approach achieves a 9.7 point improvement in the RULER score over the CacheBlend-only approach at a 124k-token input. Furthermore, TTFT is reduced by 80% at a 124k-token input when the KV caches are loaded from the SSD.

![](images/194d71ef23a9bb95cad954b861d1d37ca8e4e602b9bf92db022c90207735d611.jpg)  
Figure 1: Comparison of four KV cache concatenation strategies: (1) normal, (2) fine-tuning model, (3) KV cache recomputation, and (4) the proposed combination. By combining fine-tuned model with selective KV cache recomputation, generation quality is better than using either technique alone. Accuracy values are taken from the results on the RULER HQA shown in Figure 10.

## 2 Related Works

A variety of studies have examined how to reuse and concatenate KV caches while maintaining generation quality. Prompt Cache (Gim et al., 2024) achieves this by applying a prompt markup language to merge KV caches. PIE (He et al., 2025) addresses positional drift by aligning the concatenation point with the position expected by a model that uses a positional encoding scheme such as RoPE (Su et al., 2024). Even with positional adjustment, quality degradation remains substantial because there is no cross-attention among reused KV caches from retrieved chunks.

Prior work addressing the quality degradation caused by the absence of cross-attention can be broadly grouped into three categories. The first approach is to recompute KV caches for tokens selected according to a predefined rule. CacheBlend (Yao et al., 2025) recomputes the KV caches for token positions where the difference between the KV caches at the first layer obtained with full attention and the concatenated KV caches is large. CacheClip (Yang et al., 2025b) uses an auxiliary model to select token positions for recomputation. EPIC (Hu et al., 2025) recomputes the KV caches for a fixed number of leading tokens in each chunk because attention scores tend to concentrate on the ”attention sink” (Xiao et al., 2024) portion of the chunk when the KV caches of the chunks are precomputed. $\mathrm { A ^ { 3 } }$ (Zhou et al., 2025) uses attention scores between each question token and the document tokens to select token positions for recomputation. KVShare (Yang et al., 2025c) uses attention scores and deviations in the value entries to select token positions for recomputation not only in the prefill phase but also in the decode phase.

![](images/9d5fa1c0c56775b84df4917e4cfbbb560b939b1e2c97e07e539ecb5ef0ca4533.jpg)  
Figure 2: Evaluation results for the RULER HQA task with several approaches using Llama3.1-8B-Instruct. Although the fine-tuned-model only and recompute only approaches achieve comparable accuracy at a 4k-token input, the performance gap widens as the input context length increases. Among these approaches, the combination of Block-attention and CacheBlend (Proposed) achieves the best accuracy except for full attention.

The second approach is to fine-tune the model under the condition that cross-attention is absent between chunks so that the LLM can preserve generation quality. Block-attention (Ma et al., 2025) and TurboRAG (Lu et al., 2025) fine-tune the model so that concatenated KV caches can be reused without significant degradation. KVLink (Yang et al., 2025d) inserts trainable special tokens between chunks during model fine-tuning. These tokens have cross-attention to capture information from the preceding tokens. Apart from the fine-tuning step, KVLink is similar to EPIC.

The third approach focuses on differences in attention scores between KV caches computed with full attention and concatenated KV caches. Link0 (Hu et al., 2025) places prefix tokens before each chunk when precomputing KV caches to mitigate the effect of the attention sink. In addition to placing prefix tokens, APE (Yang et al., 2025f) computes the softmax for the context with a lower temperature and multiplies the attention scores by an another scaling factor.

![](images/7082a1aa82fc0116b593d2d5220e0ae54ec32ad203c312b030028cdd04c74a54.jpg)  
(a) 7th-layer

![](images/9dec1eea15def5f85e7bdf696087a887feb7114ac3513c89a671da8f07fd72d4.jpg)

![](images/a91a4d8271bd98834a2487812fc16e0670dbfeea06ad0868629ffe33515c904d.jpg)  
(c) 23rd-layer

(b) 15th-layer  
![](images/5befd4bb3b5994bbf690e34fc39b8f933b6aaeb8af18cebfd59428ad3c2b4c79.jpg)  
(d) 31st-layer  
Figure 3: Squared differences in the key entries between those produced by the concatenated cache approach and those obtained with full attention $( \Delta k e y ^ { 2 } )$ at layers 7 through 31 using Llama3.1-8B-Instruct (64k-token input). While the Block-attention-only approach produces a steeper distribution with a peak shifted further to the left than that of the Normal approach, the CacheBlend-only approach reduces large deviations mainly induced by special tokens for chunk separation and the attention sink that occurs within chunks. The combination of both approaches (Proposed) inherits both characteristics.

Cestola et al. (2026) reported an experimental evaluation of existing approaches. Then they proposed combining CacheBlend, Link0, and APE to achieve better accuracy. However, the combination of a fine-tuned model and other approaches has not been examined.

Although the aforementioned techniques have been evaluated on several tasks, few works have examined how response accuracy depends on input context length. Reusing KV caches for longcontext inputs can significantly benefit TTFT because the prefill recomputation cost scales as

$O ( n ^ { 2 } )$ . Existing works tend to report speed gains for long contexts but provide little empirical evidence regarding accuracy.

## 3 Methodology

## 3.1 Issues in existing approaches: response quality at long context inputs

Figure 2 shows results on a synthetic HotpotQA (HQA) (Yang et al., 2018) task generated by RULER (Hsieh et al., 2024), comparing Blockattention, CacheBlend with a 15% recomputation rate, EPIC, and KVLink. Two-token special tokens are inserted between chunks, and the KV caches at these positions are set to zero. KVLink recomputes the special tokens at the end of each chunk, whereas EPIC recomputes the 20 tokens starting from that position. To match the evaluation setup of the HQA task executed with Block-attention, we use the same prompt as Block-attention, and we use the best\_subspan\_em metric for evaluation following Liu et al. (2024). Normal denotes a model fine-tuned with the standard approach using the same dataset as Block-attention and KVLink for comparison. The input lengths are 4k, 8k, 16k, 32k, 64k, and 124k tokens, respectively. The 124k setting corresponds to 126,976 tokens, reduced from the original 128k setting (131,072 tokens) so that the final input context length after adding special tokens does not exceed the max token length of the model. At a 4k-token input, the accuracy is close to that of full attention. However, as the context length increases, the accuracy gap widens, indicating that current methods do not guarantee sufficient accuracy for very long inputs, where reusing KV caches is an effective way to reduce TTFT.

## 3.2 Motivation of proposed method: effect of fine-tuning/recomputing on the deviations in the key entries

Fine-tuning approaches such as Block-attention make the model more robust to KV cache concatenation by incorporating the assumption that there is no cross-attention between documents into the fine-tuning training data. On the other hand, recomputing methods such as CacheBlend select a set of token positions according to a predetermined rule and recompute only those KV caches. We investigate the effects of the two approaches in terms of deviations in the key entries compared with those obtained with full attention. Figure 3 shows the distribution of squared differences in the key entries between those produced by the concatenated cache approach and those obtained with full attention across three experimental conditions: (1) Normal, (2) Block-attention-only, and (3) CacheBlendonly. These distributions are obtained from the first task of the RULER HQA with 64k-token input. In our experiments, CacheBlend selects token indices based on the squared differences in the key entries, as implemented in LMCache<sup>1</sup> (Liu et al., 2025). Note that two-token special tokens are inserted between chunks, and the KV caches at these special token positions are set to zero to follow the CacheBlend implementation in vLLM (Kwon et al., 2023) + LMCache; the recomputation rate is 15%. As shown in Fig. 3, the Block-attention-only approach produces a steeper distribution with a peak shifted further to the left than the Normal approach. On the other hand, the CacheBlend-only approach reduces large deviations. The differences in the distributions arise from the fact that CacheBlend-only checks large deviations in the key entries at firstlayer, while Block-attention fine-tunes the model parameters across all layers. Combining both approaches can benefit from the strengths of each method because the two approaches contribute differently to reduce deviations in the key entries. We therefore also investigated the additional condition: (4) Block-attention+CacheBlend. As shown in Fig. 3, the Block-attention+CacheBlend approach inherits the strengths of both approaches and achieves the smallest deviations among them. These results suggest that combining Block-attention and CacheBlend can achieve higher accuracy than either method alone, as shown in Figure 1.

![](images/4176c41d872fcbd4237f447c55dfcae99a80fb1477cfe89fe219f9e2527245f4.jpg)  
Figure 4: The distribution of squared differences in the key entries with no recomputation (0%), CacheBlend (15%), and EPIC at the first layer using Llama3.1-8B-Instruct. Because EPIC recomputes KV caches only around the attention sink of each chunk, it cannot reduce large deviations far from the attention sink.

## 3.3 Other Candidates for KV Cache Recomputation and Performance Comparison

The combination of Block-attention and EPIC is also a candidate for improving performance, but the improvement may be smaller than that achieved with CacheBlend. Figure 4 shows keyvalue cache differences at the first layer using EPIC and CacheBlend. Although CacheBlend recomputes the KV caches for token positions with large deviations in the key entries, EPIC recomputes the KV caches around the attention sink of each chunk. As a result, large deviations in the key entries far from the attention sink remain unrecomputed. These deviations could cause performance degradation.

![](images/139a319cd66b3c708c530682cca15a627816ac9fec5822baf1a0e949275b16fa.jpg)  
Figure 5: Timing chart of layer-wise KV cache loading and recomputation. Loading KV caches to the CPU, transferring KV caches to the GPU and merging with position adjustment, and recomputation are executed in parallel.

Using attention scores obtained from the query, as in $\mathrm { A } ^ { \overline { { 3 } } }$ and KVShare, is also an attractive candidate for improving performance. However, our experiments using KVShare did not show any measurable improvement under long input as shown in Appendix E, so we did not include these methods from the main evaluation. Moreover, Link0 and APE could potentially improve performance, but we did not test their combination in this paper.

As shown in Fig. 2, the combination of Blockattention and CacheBlend achieves the best accuracy among the evaluated strategies. Therefore, we selected this combination for our evaluation. A finetuned model can be combined with CacheBlend’s KV cache recomputation in the same way as a nonfine-tuned model. As a result, the inference latency remains essentially the same as that of CacheBlend alone, while the accuracy improves.

## 3.4 Implementation

We implemented a layer-wise parallel operation for loading KV caches and recomputing them, as shown in Figure 5. This operation consists of three parts. First, precomputed KV caches for the selected layer are loaded from safetensors-format files on the SSD (num\_worker=8). Second, the loaded KV caches are transferred to the GPU, and the KV caches for the chunks are merged into one, followed by position adjustment of the KV caches. Finally, the transformer-layer computation, including KV cache recomputation, is performed. The first part runs with ThreadPoolExecutor, and the second and third parts use CUDA streams to overlap computation and KV cache transfer.

## 4 Experiments

This section evaluates the performance of the proposed approach. We used transformers v4.57.1 for both training and inference. First, we use Llama3.1-

8B-Instruct (Grattafiori et al., 2024) as the base model and fine-tune it to be aware of KV cache concatenation, following the training datasets for Block-attention in (Ma et al., 2025). The dataset used for fine-tuning consists of two parts. The first part consists of 40,000 samples from Tulu3 (Lambert et al., 2025). The second part is created by selecting 21,000 samples each from TriviaQA (TQA) (Joshi et al., 2017) and 2WikiMultiHopQA (2Wiki) (Ho et al., 2020). Then, ten documents are retrieved randomly for each sample using the Contriever toolkit<sup>2</sup>, and shuffling the order of samples. After that, the generated answers for these samples are produced with GPT-4o mini. Finally, samples where GPT-4o mini fails to generate an answer are excluded, and 40,000 samples are selected for the training dataset. The contexts in these datasets are split into chunks in the same way as Block-attention<sup>3</sup> to obtain the Block-attention model. Fine-tuning is performed on a dual-socket system with four NVIDIA H200 141GB GPUs and two Intel(R) Xeon(R) Platinum 8562Y+ CPUs, and inference is performed on a single GPU. The model is fine-tuned on the concatenation of these two datasets using a learning rate of $2 \times 1 0 ^ { - 6 }$ , a batch size of four (one per GPU), gradient accumulation steps of 16, one epoch, a warmup ratio of 0.03, and the AdamW optimizer. DeepSpeed<sup>4</sup> is used to accelerate the training procedure using the bfloat16 format. FlashAttention-2 (Dao, 2024) is also used for the Normal. While the Normal model is fine-tuned using the dataset only with full attention, Block-attention uses the dataset both with full attention and under the assumption that there is no cross-attention between chunks. It takes about five hours to create the Block-attention model from Llama3.1-8B-Instruct. In our experiments, KVLink uses special tokens for chunk separation as trainable special tokens during fine-tuning.

During inference, we perform the recomputation stage using FlashInfer (Ye et al., 2025) with a custom attention mask. Position adjustment for KV cache concatenation is performed in float32 while other computations in the model are performed using the bfloat16 format. We perform text generation with a single run greedy decoding (do\_sample=False, repetition\_penalty=1.0), setting max\_new\_tokens to 512 for all tasks except the RULER benchmark, where each task has its own specified value. Note that full recomputation, such as for the 0th layer, is automatically switched to causal attention mode. The prefill stage for instruction tokens and the decode stage are processed with FlashAttention-2. To facilitate future deployment with LMCache, we inserted two-token special tokens between documents and filled the KV caches corresponding to those special token indices with all-zero values. From the original RULER benchmark (Hsieh et al., 2024), all passages except those for QA tasks were concatenated into documents of no more than 512 tokens and adapted for Blockattention and CacheBlend. For QA tasks, we use the pre-existing documents directly as individual chunks.

![](images/cd98ed6d036bff78ee7fb342bd259baac237861e70a16685b3edfb6857ce3b97.jpg)  
Figure 6: Evaluation results for the RULER benchmark across several input context lengths using Llama3.1-8B-Instruct. The proposed approach shows the smallest degradation from full attention (100%) compared with approaches that use either technique alone.

![](images/afc2aabd802708fd039ac3757b0a4146c9fd00a8ad97b69af870a6e4ee4afaf2.jpg)  
Figure 7: Evaluation results for the RULER benchmark across several input context lengths using Qwen2.5-7B-Instruct. Results consistent with those for Llama3.1-8B-Instruct are obtained.

## 4.1 Generation Quality on the RULER Benchmark

Figure 6 shows the evaluation results for the RULER benchmark using Llama3.1-8B-Instruct. Tables 1 and 2 show the RULER scores of several tasks at 4k and 124k tokens. This figure demonstrates that as the number of input context tokens grows, the score gap relative to the 100% recomputation baseline (equivalent to full attention) widens. Among the evaluated variants, the Block-attention configuration with a 15% recomputation rate shows the smallest accuracy degradation, indicating that combining fine-tuning with a selective recomputation strategy improves scores in scenarios that merge KV caches.

![](images/2140905c66250b20ebe69b085502f31fabcbc83a2bfd979117ed84231e6a6a10.jpg)  
Figure 8: Evaluation results for the RULER benchmark with different numbers of documents per chunk at a 124k-token input using Llama3.1-8B-Instruct. As the number of chunks decreases, the RULER score recovers toward the full attention (100%) result. Note that the full attention result is obtained with one document per chunk.

Figure 7 reports the same set of experiments performed with Qwen2.5-7B-Instruct model (Yang et al., 2025e). This model uses YARN (Peng et al., 2024), and the learning rate is $5 \times 1 0 ^ { - 6 }$ . It takes about 7.5 hours to create the Block-attention model from Qwen2.5-7B-Instruct. The results also confirm that Block-attention combined with CacheBlend delivers the best performance with a different model.

Note that the number of chunks increases as the number of tokens grows in the experiments shown in Figs. 6 and 7. As a result, the number of chunks at a 124k-token input exceeds 200 (and 700 in the case of QA tasks). To evaluate the effect of the number of chunks alone, we also evaluate the RULER benchmark under the condition in which the number of documents per chunk is changed, reducing the number of chunks while keeping the number of tokens unchanged except for the special tokens between chunks. Figure 8 shows the impact of the number of documents per chunk on the RULER score at a 124k-token input. When the number of documents per chunk increases from 1 to 16, the score gap between full attention and the proposed approach decreases from 24.5 to 15.8. This means that increasing the number of documents per chunk can improve the RULER score, but at the cost of loading redundant chunks or lowering the retrieval hit rate because the minimum token length per chunk increases. On the other hand, the RULER score improvement of the proposed approach from the CacheBlend-only approach is larger as the number of documents per chunk is smaller. This result shows that the proposed approach is effective at maintaining generation quality even with a higher number of chunks.

<table><tr><td>Tasks</td><td>Single</td><td>MK</td><td>MV</td><td>MQ</td><td>VT</td><td>CWE</td><td>FWE</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td>Normal+0%</td><td>98.7</td><td>79.6</td><td>43.0</td><td>74.5</td><td>31.8</td><td>86.7</td><td>95.3</td><td>36.0</td><td>28.0</td><td>71.6</td></tr><tr><td>Normal+15% (CacheBlend)</td><td>100.0</td><td>92.9</td><td>56.8</td><td>84.5</td><td>53.1</td><td>97.8</td><td>98.5</td><td>74.0</td><td>48.8</td><td>84.0</td></tr><tr><td>Normal+100%</td><td>100.0</td><td>99.9</td><td>100.0</td><td>100.0</td><td>100.0</td><td>98.6</td><td>88.8</td><td>82.4</td><td>63.4</td><td>94.8</td></tr><tr><td>Block-attention+0%</td><td>99.9</td><td>93.7</td><td>53.1</td><td>89.8</td><td>51.8</td><td>86.8</td><td>95.3</td><td>69.6</td><td>47.4</td><td>82.7</td></tr><tr><td>Block-attention+15% (Proposed)</td><td>100.0</td><td>96.9</td><td>56.1</td><td>86.7</td><td>70.8</td><td>96.7</td><td>95.3</td><td>75.4</td><td>52.8</td><td>86.5</td></tr><tr><td>Block-attention+100%</td><td>100.0</td><td>99.9</td><td>89.5</td><td>100.0</td><td>100.0</td><td>98.9</td><td>85.3</td><td>81.4</td><td>61.4</td><td>93.5</td></tr></table>

Table 1: Evaluation results for the RULER benchmark at 4k-token input using Llama3.1-8B-Instruct. Single and MK denote the average score of Single NIAH 1-3, and Multi-Key NIAH 1-3, respectively. Avg. denotes the average score across all 13 tasks. At a 4k-token input, the Block-attention-only and CacheBlend-only approaches achieve scores comparable to those of the proposed approach.
<table><tr><td>Tasks</td><td>Single</td><td>MK</td><td>MV</td><td>MQ</td><td>VT</td><td>CWE</td><td>FWE</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td>Normal+0%</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>22.3</td><td>1.6</td><td>1.0</td><td>1.9</td></tr><tr><td>Normal+15% (CacheBlend)</td><td>93.3</td><td>22.1</td><td>45.8</td><td>61.5</td><td>35.0</td><td>0.0</td><td>72.7</td><td>29.4</td><td>20.0</td><td>47.0</td></tr><tr><td>Normal+100%</td><td>99.9</td><td>90.3</td><td>95.6</td><td>98.0</td><td>74.5</td><td>0.1</td><td>79.5</td><td>77.2</td><td>47.4</td><td>80.2</td></tr><tr><td>Block-attention+0%</td><td>3.2</td><td>2.1</td><td>0.7</td><td>0.6</td><td>0.0</td><td>0.0</td><td>51.9</td><td>2.4</td><td>8.8</td><td>6.2</td></tr><tr><td>Block-attention+15% (Proposed)</td><td>98.3</td><td>35.8</td><td>42.0</td><td>75.6</td><td>57.1</td><td>0.1</td><td>82.1</td><td>48.4</td><td>29.4</td><td>56.7</td></tr><tr><td>Block-attention+100%</td><td>100.0</td><td>92.1</td><td>94.9</td><td>98.2</td><td>80.9</td><td>0.1</td><td>81.7</td><td>75.0</td><td>47.8</td><td>81.1</td></tr></table>

Table 2: Evaluation results for the RULER benchmark at a 124k-token input using Llama3.1-8B-Instruct. At a 124k-token input, the proposed approach achieves the best score on most tasks compared with approaches that use either technique alone.

## 4.2 Performance in RAG and General Tasks

Table 3 shows the evaluation results for four RAG tasks. We use the same prompt as in Fig. 2. The four benchmarks are 2Wiki (Ho et al., 2020), HQA (Yang et al., 2018), Natural Questions (NQ) (Kwiatkowski et al., 2019), and TriviaQA (TQA) (Joshi et al., 2017). Ten documents are retrieved for each task using the Contriever toolkit. best\_subspan\_em is used as the evaluation metric, just as in Fig. 2. The proposed approach achieves the best accuracy among them, except for full attention. Table 4 shows the evaluation results for general-purpose tasks using Open-Compass (Contributors, 2023). As in the experiments shown for Block-attention, in-context learning (ICL) datasets such as GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), BBH (Suzgun et al., 2022), and DROP (Dua et al., 2019) are evaluated with KV cache concatenation. Zeroshot datasets such as MMLU (Hendrycks et al., 2021a), HumanEval (Chen et al., 2021), and IFEval (Zhou et al., 2023) are evaluated with full attention. For IFEval, prompt-level-strict-accuracy is selected for evaluation. As shown in Table 4, the proposed approach lies between the 0% and 100% recomputation settings, and selective recomputation is beneficial even for general tasks.

## 4.3 Token-Length Dependence of TTFT with the Proposed Approach

These results demonstrate that the combination of Block-attention and CacheBlend can substantially improve performance while reducing computational overhead. Figure 9 plots the relationship between input context length and TTFT. All plots include the running time for approximately 1.5 s to execute model.generate after KV cache preparation. In this experiment, FlashInfer with an attention mask is used for every attention computation to measure the effect of the recomputation rate. The KV caches are pre-stored on an SSD with a bandwidth of 6.5 GB/s, measured using fio. For QA tasks, the cache files corresponding to the task are loaded from the SSD. When the token length per file is short, the overhead incurred per file becomes a large fraction of the actual data read time, thereby diminishing the benefit of reusing KV cache. Accordingly, we set the number of documents per cache file to 16, resulting in an average token count of about 2.2k per file. For a context length of 4k tokens, full prefill computation yields a shorter TTFT than loading KV caches. However, once the context length reaches 8k tokens or more, loading KV caches becomes advantageous over full prefill computation, and TTFT is reduced by 80% at a 124k-token input. Figure 10 shows the relationship between TTFT and the RULER HQA accuracy under this condition. The use of a Block-attention fine-tuned model achieves an accuracy gain over the Normal model without additional TTFT cost.

<table><tr><td>Approaches</td><td>2wiki</td><td>HQA</td><td>NQ</td><td>TQA</td></tr><tr><td>Normal+0%</td><td>42.0</td><td>42.2</td><td>41.9</td><td>63.1</td></tr><tr><td>Normal+15% (CacheBlend)</td><td>65.9</td><td>66.1</td><td>57.1</td><td>73.8</td></tr><tr><td>Normal+100%</td><td>79.5</td><td>78.0</td><td>62.6</td><td>77.3</td></tr><tr><td>Block-attention+0%</td><td>68.8</td><td>70.1</td><td>57.1</td><td>74.5</td></tr><tr><td>Block-attention+15% (Proposed)</td><td>70.8</td><td>72.6</td><td>58.1</td><td>75.2</td></tr><tr><td>Block-attention+100%</td><td>78.5</td><td>77.8</td><td>61.7</td><td>77.0</td></tr></table>

Table 3: Evaluation results on four RAG benchmarks using Llama3.1-8B-Instruct. Among the approaches other than full attention, the proposed approach achieves the best accuracy.
<table><tr><td>Task Type</td><td colspan="3">General</td><td colspan="4">ICL</td></tr><tr><td>dataset setup</td><td>IFEval 0-shot</td><td>HumanEval 0-shot</td><td>MMLU 0-shot</td><td>GSM8K 4-shot</td><td>MATH 4-shot</td><td>BBH 3-shot</td><td>DROP 3shot</td></tr><tr><td>Normal+0% Normal+15% (CacheBlend)</td><td>64.3</td><td>57.3</td><td>66.0</td><td>65.1 76.3</td><td>24.3 35.9</td><td>61.9 70.6</td><td>12.4 13.3</td></tr><tr><td>Normal+100% Block-attention+0%</td><td></td><td></td><td></td><td>79.3 75.1</td><td>35.0 32.4</td><td>70.8 68.1</td><td>15.3 12.3</td></tr><tr><td>Block-attention+15% (Proposed)</td><td>66.0</td><td>64.0</td><td>65.9</td><td>78.2</td><td>35.7 34.9</td><td>70.6</td><td>12.3</td></tr><tr><td colspan="3">Block-attention+100%</td><td>79.5</td><td></td><td>70.4</td><td>13.2</td></tr></table>

Table 4: Evaluation results for seven general benchmarks using Llama3.1-8B-Instruct. Only the four ICL tasks with few-shot examples are split into different chunks. Compared with no approach, all three methods improve performance on the ICL tasks.

![](images/60ee08c684c14de493651c1909a134e09ac3258cffcb694ee2051eb49b6e1697.jpg)  
Figure 9: Averaged TTFT over 20 tasks in the RULER HQA using Llama3.1-8B-Instruct with the number of documents per cache file set to 16. The Proposed approach achieves much lower TTFT than full attention (100%).

## 5 Conclusions

We addressed the degradation in generation quality that occurs when precomputed KV caches are concatenated for long-context inputs, especially when KV cache loading becomes more advantageous than prefill computation in terms of TTFT. To reduce this performance degradation, we proposed combining KV cache-concatenation-aware finetuning with recomputation of selected KV caches. Experimental results showed that the proposed technique reduces degradation in generation quality for long-context inputs compared with either existing technique alone. We also demonstrated that TTFT is reduced by 80% at a 124k-token input when the KV caches are loaded from the SSD.

![](images/1b4d6adda69a3e81d0b9e4a9b6423d22dd955ce71ade612bac1c3212f64982b7.jpg)  
Figure 10: Relationship between averaged TTFT over 20 tasks and the accuracy in the RULER HQA at a 124k-token input using Llama3.1-8B-Instruct with the number of documents per cache file set to 16.

## Limitations

This section discusses the limitations of this paper that we intentionally leave as future work.

Limitation 1: Computation time of attention that supports recomputation for selected query indices. In our experiments, FlashInfer with an attention mask is used to compare selective recomputation with full attention under the same attention computation setting. However, causal attention can use FlashAttention-2 to further reduce computation time. If only selective recomputation uses Flash-Infer with an attention mask while full attention uses FlashAttention-2, the benefit of selective recomputation plus KV cache loading decreases as shown in Appendix C. However, selective recomputation without an attention mask incurs attention to future KV positions, which results in quality degradation, especially for long-context inputs. To fully exploit selective recomputation, a faster attention mechanism that supports selected query indices is required.

Limitation 2: Evaluation of attention-scorebased recomputation. In our experiments, we did not evaluate attention-score-based recomputation methods such as $\mathrm { A ^ { 3 } }$ and KVShare. However, the use of these recomputation methods should also be evaluated because these methods could achieve better performance than CacheBlend. While the evaluation results for the RULER benchmark using KVShare as shown in Appendix E, detailed evaluation is required.

Limitation 3: Evaluation of an additional combination of Link0 and APE. We did not evaluate the combination of Link0 and APE because we focused on the combination of KV cache recomputation and model fine-tuning. However, as stated in Cestola et al.(2026), an additional combination of Link0 and APE could help achieve even better performance.

Limitation 4: Fine-tuning thinking models. In this paper, we did not fine-tune thinking models such as Qwen3 (Yang et al., 2025a) and gpt-oss (OpenAI et al., 2025). A new fine-tuning strategy that incorporates a Block-attention-style design without reducing chain-of-thought capability is required.

## References

Samuel Cestola, Tianxiang Xia, Zheng Weiyan, Zheng Pengfei, and Diego Didona. 2026. An experimen-

tal study of kv cache reuse strategies in chunk-level caching systems. Preprint, arXiv:2603.20218.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models.

Tri Dao. 2024. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, Minneapolis, Minnesota. Association for Computational Linguistics.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-augmented generation for large language models: A survey. Preprint, arXiv:2312.10997.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. 2024. Prompt cache: Modular attention reuse for low-latency inference. In Proceedings of Machine Learning and Systems, volume 6, pages 325–338.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Zhenyu He, Jun Zhang, Shengjie Luo, Jingjing Xu, Zhi Zhang, and Di He. 2025. Let the code llm edit itself when you edit the code. In International Conference on Learning Representations, volume 2025, pages 59637–59653.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the MATH dataset. In Thirtyfifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. 2024. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling.

Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, tiancheng hu, zhang qin, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. 2025. EPIC: Efficient position-independent caching for serving large language models. In Forty-second International Conference on Machine Learning.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions ofthe Associationfor Computational Linguistics, 7:452–466.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, SOSP ’23.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le

Bras, Oyvind Tafjord, Christopher Wilhelm, Luca Soldaini, and 4 others. 2025. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling.

Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu. 2022. A survey on retrieval-augmented text generation. Preprint, arXiv:2202.01110.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Yuhan Liu, Yihua Cheng, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng, Yuyang Huang, Samuel Shen, Rui Zhang, Kuntai Du, and Junchen Jiang. 2025. Lmcache: An efficient kv cache layer for enterprisescale llm inference. Preprint, arXiv:2510.09665.

Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang. 2025. TurboRAG: Accelerating retrieval-augmented generation with precomputed KV caches for chunked text. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6588–6601, Suzhou, China. Association for Computational Linguistics.

Dongyang Ma, Yan Wang, and Tian Lan. 2025. Blockattention for efficient prefilling. In The Thirteenth International Conference on Learning Representations.

OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, and 108 others. 2025. gpt-oss-120b & gptoss-20b model card. Preprint, arXiv:2508.10925.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. YaRN: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. Preprint, arXiv:2210.09261.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

Bin Yang, Qiuyu Leng, Jun Zeng, and Zhenhua Wu. 2025b. Cacheclip: Accelerating rag with effective kv cache reuse. Preprint, arXiv:2510.10129.

Huan Yang, Renji Zhang, Mingzhe Huang, Weijun Wang, Yin Tang, Yuanchun Li, Yunxin Liu, and Deyu Zhang. 2025c. Kvshare: An llm service system with efficient and effective multi-tenant kv cache reuse. Preprint, arXiv:2503.16525.

Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025d. KVLink: Accelerating large language models via efficient KV cache reuse. In The Thirtyninth Annual Conference on Neural Information Processing Systems.

Qwen: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025e. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Xinyu Yang, Tianqi Chen, and Beidi Chen. 2025f. Ape: Faster and longer context-augmented generation via adaptive parallel encoding. In International Conference on Learning Representations, volume 2025, pages 82887–82907.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. Cacheblend: Fast large language model serving for rag with cached knowledge fusion. In Twentieth European Conference on Computer Systems, pages 94–109.

Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. Flashinfer: Efficient and customizable attention engine for llm inference serving. Preprint, arXiv:2501.01005.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. Sglang: Efficient execution of structured language model programs. In

Advances in Neural Information Processing Systems, volume 37, pages 62557–62583. Curran Associates, Inc.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. Preprint, arXiv:2311.07911.

Yuechi Zhou, Yi Su, Jianxin Zhang, Juntao Li, Qingrong Xia, Zhefeng Wang, Xinyu Duan, and Baoxing Huai. 2025. A<sup>3</sup>: Attention-aware accurate kv cache fusion for fast large language model serving. Preprint, arXiv:2511.17560.

## A Detailed Results for NIAH tasks

Tables 5, 6, 7, and 8 are detailed results for the NIAH tasks shown in Tables 1 and 2. S1-S3 denote the Single NIAH 1-3 tasks, and M1-M3 denote the Multi-key NIAH 1-3 tasks.

<table><tr><td rowspan="2">Tasks</td><td colspan="3">Single NIAH</td></tr><tr><td>S1</td><td>S2</td><td>S3</td></tr><tr><td>Normal+0%</td><td>100.0</td><td>98.4</td><td>97.6</td></tr><tr><td>Normal+15% (CacheBlend)</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Normal+100%</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Block-attention+0%</td><td>100.0</td><td>99.8</td><td>100.0</td></tr><tr><td>Block-attention+15%</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>(Proposed) Block-attention+100%</td><td>100.0</td><td>100.0</td><td>100.0</td></tr></table>

Table 5: Evaluation results for the Single NIAH of the RULER benchmark at a 4k-token input using Llama3.1- 8B-Instruct.

<table><tr><td rowspan="2">Tasks</td><td colspan="3">Multi-key NIAH</td></tr><tr><td>M1</td><td>M2</td><td>M3</td></tr><tr><td>Normal+0% Normal+15%</td><td>70.2</td><td>88.6</td><td>80.0</td></tr><tr><td>(CacheBlend)</td><td>91.0</td><td>95.8</td><td>91.8</td></tr><tr><td>Normal+100%</td><td>100.0</td><td>100.0</td><td>99.8</td></tr><tr><td>Block-attention+0%</td><td>93.0</td><td>98.8</td><td>89.2</td></tr><tr><td>Block-attention+15%</td><td>95.8</td><td>98.4</td><td></td></tr><tr><td>(Proposed) Block-attention+100%</td><td>100.0</td><td>100.0</td><td>96.6 99.8</td></tr></table>

Table 6: Evaluation results for the Multi-key NIAH of the RULER benchmark at a 4k-token input using Llama3.1-8B-Instruct.

## B Data Proportions

Table 9 shows the number of training samples used to train the Normal model and the Block-attention model, respectively. Note that samples from the first part that have code-related context are used only for full attention. As a result, the number of training samples for Block-attention is smaller than that for Normal/Block-attention. Table 10 shows the number of evaluation samples for the four RAG benchmarks. Tables 11 and 12 show the number of chunks for the RULER benchmark using Llama3.1- 8B-instruct and Qwen2.5-7B-Instruct, respectively.

<table><tr><td rowspan="2">Tasks</td><td colspan="3">Single NIAH</td></tr><tr><td>S1</td><td>S2</td><td>S3</td></tr><tr><td>Normal+0%</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Normal+15% (CacheBlend)</td><td>99.2</td><td>83.2</td><td>97.4</td></tr><tr><td>Normal+100%</td><td>100.0</td><td>100.0</td><td>99.6</td></tr><tr><td>Block-attention+0%</td><td>3.2</td><td>4.4</td><td>2.0</td></tr><tr><td>Block-attention+15%</td><td>100.0</td><td>95.0</td><td>99.8</td></tr><tr><td>(Proposed) Block-attention+100%</td><td>100.0</td><td>100.0</td><td>100.0</td></tr></table>

Table 7: Evaluation results for the Single NIAH of the RULER benchmark at a 124k-token input using Llama3.1-8B-Instruct.

<table><tr><td>Tasks</td><td>Multi-key NIAH M1 M2 M3</td></tr><tr><td>Normal+0%</td><td>0.0 0.0 0.0</td></tr><tr><td>Normal+15% 51.6</td><td>14.8 0.0</td></tr><tr><td>(CacheBlend) Normal+100% 97.8</td><td>92.2 81.0</td></tr><tr><td>Block-attention+0%</td><td>6.4 0.0 0.0</td></tr><tr><td>Block-attention+15%</td><td>31.4 2.2</td></tr><tr><td>(Proposed) Block-attention+100% 99.0</td><td>73.8 92.2 85.0</td></tr></table>

Table 8: Evaluation results for the Multi-key NIAH of the RULER benchmark at a 124k-token input using Llama3.1-8B-Instruct.

<table><tr><td>Dataset</td><td>Normal and Block-attention</td><td>Block-attention</td></tr><tr><td>Tulu3</td><td>40,000</td><td>33,532</td></tr><tr><td>2wiki</td><td>~20,000</td><td>~20,000</td></tr><tr><td>TQA</td><td>~20,000</td><td>~20,000</td></tr></table>

Table 9: Quantities of data used during model finetuning.

## C Comparison of TTFT between Flashinfer and FlashAttention-2

Figure 11 plots the relationship between input context length and TTFT. In addition to the results in Fig. 9, TTFT of full attention with FlashAttention-2 is included. Because FlashAttention-2 reduces computation time, TTFT of full attention with FlashAttention-2 is comparable to that of 15% recomputation with Flashinfer.

<table><tr><td>Dataset</td><td>Quantity</td></tr><tr><td>2wiki</td><td>12576</td></tr><tr><td>HQA</td><td>7405</td></tr><tr><td>NQ</td><td>3610</td></tr><tr><td>TQA</td><td>11313</td></tr></table>

Table 10: Quantities of Document QA data.
<table><tr><td>Tasks</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>124k</td></tr><tr><td>S1</td><td>8.0</td><td>16.0</td><td>33.0</td><td>65.0</td><td>130.0</td><td>252.0</td></tr><tr><td>S2</td><td>7.4</td><td>16.1</td><td>32.8</td><td>65.8</td><td>132.4</td><td>256.8</td></tr><tr><td>S3</td><td>7.6</td><td>15.5</td><td>32.3</td><td>65.8</td><td>132.6</td><td>256.6</td></tr><tr><td>M1</td><td>7.4</td><td>16.1</td><td>32.6</td><td>65.8</td><td>132.3</td><td>256.8</td></tr><tr><td>M2</td><td>7.9</td><td>15.8</td><td>32.6</td><td>64.8</td><td>130.0</td><td>252.3</td></tr><tr><td>M3</td><td>8.4</td><td>16.4</td><td>33.1</td><td>67.7</td><td>137.3</td><td>268.0</td></tr><tr><td>MQ</td><td>7.4</td><td>15.9</td><td>32.8</td><td>66.0</td><td>132.6</td><td>256.5</td></tr><tr><td>MV</td><td>7.6</td><td>15.9</td><td>32.6</td><td>65.2</td><td>132.5</td><td>256.9</td></tr><tr><td>VT</td><td>8.2</td><td>17.0</td><td>33.0</td><td>65.0</td><td>130.0</td><td>252.0</td></tr><tr><td>CWE</td><td>8.0</td><td>16.0</td><td>32.0</td><td>64.2</td><td>128.7</td><td>249.0</td></tr><tr><td>FWE</td><td>7.9</td><td>14.8</td><td>29.5</td><td>61.6</td><td>124.9</td><td>241.8</td></tr><tr><td>QA1</td><td>25.4</td><td>52.2</td><td>101.5</td><td>197.5</td><td>385.5</td><td>749.5</td></tr><tr><td>QA2</td><td>24.3</td><td>55.9</td><td>118.4</td><td>243.1</td><td>484.4</td><td>944.2</td></tr></table>

Table 11: Average number of chunks for the RULER benchmark using Llama3.1-8B-Instruct. Note that the number of chunks for the RULER HQA is slightly different from that for QA2.

## D Effect of the overhead incurred per file on TTFT

Figure 12 plots the relationship between input context length and TTFT when the number of documents per cache file is one.

TTFT of Block-attention is comparable to that of the proposed approach because a large number of KV cache files adds overhead beyond the I/O bandwidth-limited read time. In this case, the context length needs to be 32k tokens or more to take advantage of loading KV caches. Figure 13 shows the relationship between TTFT and the RULER HQA accuracy under this condition.

## E RULER benchmark results using KVShare

We evaluated KVShare, one of the attentionscore-based recomputing methods. Note that KVShare was employed only in the prefill stage. Figures 14 and 15 show the evaluation results for the RULER benchmark using Llama3.1-8B-Instruct and Qwen2.5-7B-Instruct, respectively. Although the combination of fine-tuned model and recomputing method is still effective in KVShare, CacheBlend achieves better score at 124k-token input.

<table><tr><td>Tasks</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>124k</td></tr><tr><td>S1</td><td>8.0</td><td>16.0</td><td>33.0</td><td>65.0</td><td>130.0</td><td>252.0</td></tr><tr><td>S2</td><td>7.8</td><td>15.1</td><td>32.8</td><td>65.9</td><td>132.7</td><td>256.8</td></tr><tr><td>S3</td><td>7.6</td><td>16.0</td><td>32.4</td><td>65.6</td><td>132.3</td><td>256.6</td></tr><tr><td>M1</td><td>7.3</td><td>16.0</td><td>32.5</td><td>65.3</td><td>132.3</td><td>256.8</td></tr><tr><td>M2</td><td>7.9</td><td>15.8</td><td>32.3</td><td>64.8</td><td>130.6</td><td>252.8</td></tr><tr><td>M3</td><td>9.0</td><td>17.8</td><td>35.3</td><td>68.7</td><td>141.4</td><td>277.6</td></tr><tr><td>MQ</td><td>7.9</td><td>15.8</td><td>32.9</td><td>65.8</td><td>132.9</td><td>256.4</td></tr><tr><td>MV</td><td>7.7</td><td>15.6</td><td>32.4</td><td>65.6</td><td>132.6</td><td>256.8</td></tr><tr><td>VT</td><td>8.2</td><td>17.2</td><td>33.2</td><td>65.2</td><td>130.2</td><td>252.2</td></tr><tr><td>CWE</td><td>8.7</td><td>16.8</td><td>32.8</td><td>64.9</td><td>129.2</td><td>250.0</td></tr><tr><td>FWE</td><td>7.9</td><td>14.9</td><td>29.6</td><td>61.7</td><td>124.8</td><td>241.9</td></tr><tr><td>QA1</td><td>24.9</td><td>50.8</td><td>98.2</td><td>190.0</td><td>374.2</td><td>721.8</td></tr><tr><td>QA2</td><td>23.2</td><td>52.7</td><td>112.1</td><td>228.9</td><td>459.1</td><td>896.7</td></tr></table>

Table 12: Average number of chunks for the RULER benchmark using Qwen2.5-7B-Instruct.

![](images/8f19179afdfcb23e4a15be244da0c6c1787b387436799f6cb3a3a024e55b7ed6.jpg)  
Figure 11: Averaged TTFT over 20 tasks in the RULER HQA using Llama3.1-8B-Instruct with the number of documents per cache file set to 16. TTFT of full attention with FlashAttention-2 (FA2) is comparable to that of 15% recomputation with Flashinfer.

## F Evaluation in the RULER HQA and four RAG benchmarks

We used the input context shown below for the RULER HQA and four RAG benchmarks, as shown in Figure 16. The number of chunks in the RULER HQA in Fig. 2 is slightly different from that in QA2 in the RULER benchmark because of the slightly different handling of the QA task.

## G Use of AI Assistants

We used AI assistants for paraphrasing support during the writing of the manuscript. Moreover, as mentioned in Section 4, the generated answers for the training dataset were produced with GPT-4o mini and compiled into a dataset.

![](images/cdb2e4a403182717bdded43d154b31f7ae0748a0ee5fc56ccf9949fddf71fd94.jpg)  
Figure 12: Averaged TTFT over 20 tasks in the RULER HQA using Llama3.1-8B-Instruct with the number of documents per cache file set to one.

![](images/ba3d0359061b980c5ce3f9c05d9188d5fbc2b78b282b641d1cc0cdf150ab2ade.jpg)  
Figure 13: Relationship between averaged TTFT over 20 tasks and the accuracy in the RULER HQA at a 124k-token input using Llama3.1-8B-Instruct with the number of documents per cache file to one.

![](images/405377dd039b1fffc8567dd4b375b6e61daf8cb9ab52216d761a8e16020895bb.jpg)  
Figure 14: Evaluation results for the RULER benchmark across several input context lengths using Llama3.1-8B-Instruct. Combination of fine-tuned model and KVShare also achieves better score than KVShare alone.

![](images/66ff5e684ff1e873e5e3007518aba9554f355e32832b2f719fab2d74f2443c49.jpg)  
Figure 15: Evaluation results for the RULER benchmark across several input context lengths using Qwen2.5-7B-Instruct. Combination of fine-tuned model and KVShare also achieves better score than KVShare alone.

![](images/21a7409f37fbc93343a9173c57c32fc5666356a469c7e7bfafed66d09c0a40a8.jpg)  
Figure 16: Prompt example used in the RULER HQA and four RAG benchmarks.