# Faithfulness Is Not Free: Auditing Offline KV-Cache Quantization in Retrieval-Augmented Generation

Atta Ul Asad<sup>1\*</sup> Ahsan Bilal<sup>2\*</sup> Muhammad Ali<sup>3</sup> Muhammad Haseeb<sup>4</sup> Dean F. Hougen<sup>2</sup>

<sup>1</sup>Lahore University of Management Sciences (LUMS) <sup>2</sup>University of Oklahoma

<sup>3</sup>Air University <sup>4</sup>National University of Sciences and Technology (NUST)

26280099@lums.edu.pk {ahsan.bilal-1, hougen}@ou.edu

## Abstract

Retrieval-augmented generation systems can precompute and store key-value caches of retrieved documents to avoid re-encoding context at every query. Quantizing these caches further reduces storage, but no prior work asks whether compression damagesfaithfulness, whether responses remain grounded in the retrieved evidence. Faithfulness and accuracy are not equivalent: a model can produce a correct answer that is no longer supported by the context it was given. We evaluate Qwen2.5-7B-Instruct under INT8 and INT4 quantization on RGB and HotpotQA, measuring both accuracy and faithfulness with a hallucination detector, NLI entailment, and an LLM judge. INT8 is nearlossless across both metrics. INT4 reduces accuracy and, more critically, even among answers that remain factually correct, over 90% of faithfulness changes are negative, i.e., accuracy metrics are blind to this regression. The harm grows under noisy retrieval and with more retrieved chunks. Faithfulness must be audited before compressed caches are deployed.

## 1 Introduction

Retrieval-augmented generation (RAG) grounds model responses in external documents, but reprocessing the retrieved context for every query is expensive. Recent systems eliminate this cost by precomputing key-value (KV) caches for documents once and reusing them at inference time (Lu et al., 2025; Yao et al., 2025). As these offline caches scale with corpus size and retrieval depth, their storage footprint becomes a bottleneck, making low-bit quantization an attractive solution.

Prior work shows KV caches can be quantized to INT8 and INT4 with little accuracy degradation (Liu et al., 2024; Hooper et al., 2024), though aggressive compression can harm reasoning on complex tasks (Liu et al., 2025). However, all existing evaluations measure only task accuracy. They leave open a more subtle question: do quantized caches preserve faithfulness, whether generated responses remain supported by the retrieved evidence? This distinction matters. Exact Match and F1 measure agreement with a reference answer; they do not verify whether the answer is grounded in the provided context. A model can preserve the correct answer string while becoming less dependent on retrieved evidence, creating a hidden failure mode in compressed RAG systems (Yen et al., 2025; Tamber et al., 2025).

To our knowledge, this is the first audit of offline KV-cache quantization in RAG that evaluates faithfulness, showing that cache precision can affect grounding even when answer accuracy is preserved. Figure 1 shows our controlled protocol, which isolates cache precision as the sole variable and evaluates faithfulness with three independent signals. Our contributions are threefold. First, we introduce a faithfulness-centered audit protocol that avoids positional-stitching confounds by caching the full retrieved context in a single causal pass. Second, we show that INT8 is near-lossless, while INT4 silently degrades grounding even on accuracy-preserved examples (>90% of faithfulness flips are negative; McNemar $p \ < \ 1 0 ^ { - 2 0 } )$ . Third, we show that this degradation is amplified by retrieval noise and larger retrieval depth, establishing faithfulness (not accuracy alone) as the necessary audit signal for compressed RAG.

## 2 Background and Related Work

## 2.1 KV-cache compression.

KIVI (Liu et al., 2024) achieves near-lossless 2-bit asymmetric quantization; KVQuant (Hooper et al., 2024) reaches sub-4-bit via per-channel key quantization; ZipCache (He et al., 2024) identifies tokenlevel precision sensitivity; KVTuner (Li et al., 2025) sets 4-bit as the safe floor for Qwen2.5-7B on reasoning tasks. Sub-4-bit can cause up to 59% accuracy loss on long-context tasks (Mekala et al., 2025), prompting variance-normalized quantization (Müller et al., 2026) and query-aware mixedprecision allocation (Zhang et al., 2025). Gap: all of these evaluate online cache compression via accuracy or perplexity; none audits faithfulness in retrieval-augmented settings.

![](images/e325d07241d6f30360cb5840de9b30fb3c72c88534aa9c75af84900ea5e32002.jpg)  
Figure 1: Audit pipeline. Prepare: retrieved documents are prefilled in one causal pass, and the resulting KV cache is quantized and stored. Inference: the stored cache is dequantized and reused for each query. Evaluate: three complementary faithfulness signals (HHEM hallucination rate ↓, DeBERTa-v3 NLI entailment ↑, LLM-as-judge ↑) are measured alongside standard accuracy metrics.

## 2.2 RAG faithfulness.

RAGChecker (Ru et al., 2024) and RAGTruth (Niu et al., 2024) show that EM and F1 miss systematic faithfulness failures. HELMET (Yen et al., 2025) finds standard accuracy proxies correlate poorly with downstream RAG quality. LLM-as-judge outperforms detector-only methods for faithfulness evaluation (Tamber et al., 2025; Song et al., 2025). Distractor passages cause systematic grounding failures even without compression (Wu et al., 2025; Amiraz et al., 2025). Gap: none of these studies examines whether cache compression introduces faithfulness failures beyond those from retrieval noise alone.

## 2.3 Offline KV-cache RAG.

TurboRAG (Lu et al., 2025) stores chunklevel caches offline to eliminate repeated prefill; CacheBlend (Yao et al., 2025) addresses per-chunk positional mismatch via selective recomputation, reducing TTFT by 2.2–3.3×. Gap: neither examines whether compressing stored caches harms faithfulness.

## 3 Method

Our goal is to isolate the effect of KV-cache quantization in RAG. We keep the retrieved evidence, prompt, and decoding fixed, and vary only the stored cache precision. The method builds one position-consistent cache for the full retrieved context, quantizes it before storage, dequantizes it before decoding, and compares BF16, INT8, and INT4 under identical inference conditions.

Building a unified cache. For each query, we concatenate the system prompt with the top-K retrieved chunks and run one causal prefill pass. The resulting KV cache is a single unified cache for the full retrieved context, and this is the cache we compress. We avoid caching chunks independently and stitching them later, because independent chunk caches are created at local token positions but consumed at global positions during generation. This positional mismatch can degrade output quality by itself. Using a single prefill pass removes this confound and ensures that cache bit-width is the only variable changed across conditions.

Quantizing the cache. After prefill, we quantize the KV cache before storage and dequantize it back to bfloat16 before decoding. This matches an offline RAG deployment in which document caches are written to disk once and reused at inference time. INT8 uses per-token asymmetric quantization (Liu et al., 2024). INT4 uses group-wise quantization with groups of 64 channels, where each group stores its own scale and zero-point. This reduces the risk that large-magnitude channels dominate the quantization range and collapse smaller values under INT4’s 16 levels (Liu et al., 2024; Hooper et al., 2024). The on-disk footprint of one KV cache is

$$
\begin{array} { r } { M ( b , g , o ) = 2 L H d _ { h } N \cdot \frac { b } { 8 } + \frac { 2 L H d _ { h } N } { g } \cdot \frac { o } { 8 } , } \end{array}\tag{1}
$$

where L, H, $d _ { h }$ , and N denote the number of layers, KV heads, head dimension, and cached tokens. The quantization bit-width is b, the group size is g, and o is the scale/zero-point overhead per group. BF16 (b=16) has no overhead term. INT8 uses b=8 and $g { = } d _ { h }$ , while INT4 uses b=4 and $g { = } 6 4$ Because INT8 and INT4 store metadata for each group, their realized compression ratios are lower than the nominal 2× and 4× limits.

Precision comparisons. Using the same retrieved chunks and greedy decoding, we compare four cache settings. C0 (Oracle) runs standard fullcontext generation without a cache round trip. C1 (BF16) stores and reloads the uncompressed cache, serving as the cache baseline. C2 (INT8) and C3 (INT4) store quantized caches and dequantize them before decoding. Thus, C1–C2 isolates the effect of INT8 compression, while C1–C3 isolates the effect of aggressive INT4 compression.

## 4 Experimental Setup

We evaluate Qwen2.5-7B-Instruct (Yang et al., 2024), a bfloat16-native 7B instruction-tuned decoder with a 32K-token context window. Its native bfloat16 precision avoids the float16 dynamicrange overflow that silently corrupts cached KV values on this architecture.

We use two complementary benchmarks. RGB (Chen et al., 2024) (300 examples) provides its own positive and distractor passages, keeping retrieval recall high and the noise structure controlled. HotpotQA (Yang et al., 2018) (300 examples, distractor split) requires combining evidence across at least two supporting passages mixed with distractors, testing multi-hop reasoning. Each example is evaluated at $K \in \{ 1 , 3 , 5 \}$ retrieved chunks, giving $3 0 0 \times 3 \times 4 = 3 . 6 0 0$ generations per dataset. Retrieval uses a FAISS index (Johnson et al., 2021) over bge-small-en-v1.5 (BAAI, 2023) embeddings; we retrieve the top-10 chunks and slice to the first K.

We report both answer accuracy and evidence faithfulness. Accuracy is measured by containment exact match (EM), which marks a prediction correct if any normalized gold alias appears in the generated answer, and by token-level F1 as a secondary overlap metric. Faithfulness is measured with three independent signals: HHEM-2.1-Open (Vectara, 2024), reported as hallucination rate at threshold 0.5 (↓); DeBERTa-v3-large NLI entailment (He et al., 2023) (↑); and an LLM judge using Claude Haiku 4.5 (↑). Since HHEM and NLI agree only moderately on RGB (Pearson r=0.44), we treat them as complementary diagnostics. The LLM judge achieves 92% agreement on 50 handchecked labels and serves as the primary faithfulness signal. We test three hypotheses using C1 BF16 as the uncompressed reference and C3 INT4 as the aggressive compression condition. For H1, we test whether faithfulness degrades even when accuracy is preserved. We restrict to examples whose containment-EM outcome is unchanged between C1 and C3, then apply a paired McNemar test to faithfulness label flips. For H2, we test whether harm increases with retrieval depth by fitting a linear slope over $K \in \{ 1 , 3 , 5 \}$ . Because only three retrieval depths are available, this test is underpowered $\scriptstyle ( p = 0 . 1 2 )$ and is reported as directional. For H3, we test whether retrieval noise amplifies INT4 degradation by regressing the per-example INT4 faithfulness gap on distractor fraction within each K. H3 is evaluated only on RGB, where perexample noise ratios are available.

## 5 Results

Figure 2 reports the complete results across RGB and HotpotQA, and Table 1 summarizes the main effects.

Accuracy and hidden faithfulness harm. INT8 remains close to the BF16 cache baseline across accuracy and faithfulness metrics, whereas INT4 consistently degrades performance. INT4 reduces containment-EM (as shown in Figs. 2a, 2e) and token-F1 (as shown in Figs. 2b, 2f); correct→wrong flips greatly outnumber wrong→correct flips on both RGB (133 vs. 10) and HotpotQA (117 vs. 24; Fig. 2k). More importantly, INT4 also causes faithfulness loss that accuracy does not expose. On the accuracypreserved subset, where containment-EM is unchanged between C1 and C3, faithfulness flips remain overwhelmingly negative: the LLM judge records 231 worsenings vs. 24 improvements on RGB $( p < 1 0 ^ { - 3 8 } )$ and 173 vs. 31 on HotpotQA (p<10<sup>−30</sup>; Fig. 2k). HHEM corroborates this trend on RGB for all K and on HotpotQA at K=5. Thus, INT4 does not merely make more answers wrong; it also makes some still-correct answers less grounded in the retrieved evidence. On HotpotQA at low K, INT4 additionally disrupts refusal calibration: the refusal rate drops from 0.513 to 0.317 at K=1 (Fig. 2i), which artificially inflates containment-EM and NLI entailment (the † markers in Figs. 2e, 2h). For this reason, the LLM judge is the primary H1 signal on HotpotQA.

Amplification, storage, and degeneration. The degradation becomes larger under harder retrieval settings. The INT4–BF16 hallucination gap increases monotonically from K=1 to 5 on both benchmarks (Figs. 2c, 2g), although the slope test over only three retrieval depths is underpowered (p=0.12), so we treat H2 as suggestive. On RGB, where per-example noise ratios are available, the INT4 faithfulness gap also grows with distractor fraction independently of retrieval depth (β=0.22, significant within-K at K=3 and K=5), supporting H3. These failures occur despite substantial storage savings: at K=5, INT8 and INT4 reduce cache size by approximately 1.9× and 3.6×, respectively (Fig. 2j), with INT4 falling below the nominal 4× due to per-block scale and zero-point overhead in Eq. 1. INT4 is also the only condition that produces degenerate generations, i.e., empty outputs or repetition loops reaching 6% on RGB

![](images/6bddb65a47cd60c07917e1463af5d4a10ba2e6c32ec8b24d4186d8a3916eccf6.jpg)  
(a) RGB EM accuracy

![](images/e9c7e0209d898b49f4b0603efdc81d02f8b6a0e0c9b442a0ae6574adb9d973b1.jpg)  
(b) RGB F1

![](images/84f3b1f7bb0f4f91c3a00a2dfdc78caca35563cec8086c22106753f69d01c9a1.jpg)

![](images/477c6d20d0ca09c457ce6a04b157c41d36398461645d715e8b26aa6f574451e1.jpg)  
(c) RGB hallucination  
(d) RGB entailment

![](images/8c7e9ca842325aea3e2069eb2963fc0e3f67933a583c7d9bc79299a2f1c31735.jpg)  
(e) HotpotQA EM accuracy

![](images/42d50527eff742321c22bba6025766c48f7717fe0e616f28040f209b20993360.jpg)  
(f) HotpotQA F1

![](images/1539a9d75b2e6985c3b51ec0c9daab06ad00a77f3a8e1ec0c3466d69f919a54e.jpg)  
(g) HotpotQA hallucination

![](images/1921ae2447122c3c4e8eae4d22707eef2bc66357be72213285e1c5ed3d804734.jpg)  
(h) HotpotQA entailment

![](images/b23dd3920f99f96bacc4447e2438b59fa999c13f3a1a00fc49f530d66fd194e4.jpg)  
(i) HotpotQA refusal

![](images/85ca9a9057873696d39d7ca3cf447cca36809ac832fa1b9645a738412c7cca95.jpg)  
(j) KV cache size

![](images/10766cf9f2e6fc38fbaf8c2996b079a92fd4ce895eb4d82f58918d2fd63b118a.jpg)  
(k) Hidden faithfulness harm (H1)

![](images/7616a1a8f1405110c16c9884b721ee80acc2d855b0c2b46f6d4fa7c476a5b929.jpg)  
(l) Degeneracy (K=5)  
Figure 2: Benchmark results across RGB and HotpotQA under four precision conditions: C0–C3. † marks the HotpotQA K=1 refusal artifact: INT4 under-refuses, which inflates surface EM and NLI entailment—not a genuine gain. In (j), tick labels are dataset,K (R=RGB, H=HotpotQA); the inset shows on-disk shrinkage vs. C1 BF16: ∼1.9× (INT8) and ∼3.6× (INT4) at K=5, below the nominal 4× because INT4 stores per-block scale/zero-point pairs. In (k), columns group by dataset; within each, EM=containment exact-match, HH=HHEM, LJ=LLM-judge; bars are McNemar flip counts b (worsened) vs. c (improved). Significance: $\ast \ast \ast p { < } 1 0 ^ { - 3 }$ , ∗∗ $\scriptstyle : p < 1 0 ^ { - 2 }$ , ∗ p<0.05.

and 3% on HotpotQA at K=5 (as shown in Fig. 2l);   
BF16 and INT8 never degenerate.

## 6 Conclusion

KV-cache compression should be treated as a reliability decision, not only a storage optimization. In offline RAG, the cached KV forms the model’s effective interface to retrieved evidence; when this representation is aggressively compressed, the model can preserve the surface answer while weakening its dependence on the supporting context. The central lesson is that compressed RAG systems cannot be validated by EM or F1 alone. Before deploying low-bit offline caches, systems should audit whether answers remain grounded in the retrieved documents. Faithfulness is therefore a necessary evaluation target for practical KV-cache compression, especially when storage savings are obtained through aggressive quantization.

## Limitations

This study isolates KV-cache precision under a controlled offline RAG setup, but its scope is limited to one model family, two QA-style benchmarks, and three retrieval depths. The magnitude of faithfulness loss may differ for larger models, other architectures, denser retrieval settings, or production retrievers. HotpotQA also introduces a refusalcalibration confound at low K, which limits the reliability of HHEM and NLI in that regime; we therefore rely on the LLM judge as the primary signal in that regime.

## References

Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. 2025. The distracting effect: Understanding irrelevant passages in RAG. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18228–18258, Vienna, Austria. Association for Computational Linguistics.

BAAI. 2023. BGE-small-en-v1.5: BAAI general embedding. https://huggingface.co/BAAI/ bge-small-en-v1.5.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 17754–17762.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2023. DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradientdisentangled embedding sharing. In The Eleventh International Conference on Learning Representations (ICLR 2023).

Yefei He, Luoming Zhang, Weijia Wu, Jing Liu, Hong Zhou, and Bohan Zhuang. 2024. ZipCache: Accurate and efficient KV cache quantization with salient token identification. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), pages 68287–68307.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. 2024. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), pages 1270–1303.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547.

Xing Li, Zeyu Xing, Yiming Li, Linping Qu, Hui-Ling Zhen, Yiwu Yao, Wulong Liu, Sinno Jialin Pan, and Mingxuan Yuan. 2025. KVTuner: Sensitivity-aware layer-wise mixed-precision KV cache quantization for efficient and nearly lossless LLM inference. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 36451–36485. PMLR.

Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, and Lu Hou. 2025. Quantization hurts reasoning? An empirical study on quantized reasoning models. In Second Conference on Language Modeling (COLM 2025).

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32332–32344. PMLR.

Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang. 2025. TurboRAG: Accelerating retrieval-augmented generation with precomputed KV caches for chunked text. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6588–6601. Association for Computational Linguistics.

Anmol Reddy Mekala, Anirudh Atmakuru, Yixiao Song, Marzena Karpinska, and Mohit Iyyer. 2025. Does quantization affect models’ performance on longcontext tasks? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9422–9470, Suzhou, China. Association for Computational Linguistics.

Lorenz K. Müller, Philippe Bich, Chiara Boretti, Hyun-Min Chang, Jiawei Zhuang, and Lukas Cavigelli. 2026. KVarN: Variance-normalized KV-cache quantization mitigates error accumulation in reasoning tasks. Preprint, arXiv:2606.03458. Preprint.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2024. RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10862– 10878, Bangkok, Thailand. Association for Computational Linguistics.

Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024.

RAGChecker: A fine-grained framework for diagnosing retrieval-augmented generation. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), pages 21999–22027.

Maojia Song, Shang Hong Sim, Rishabh Bhardwaj, Hai Leong Chieu, Navonil Majumder, and Soujanya Poria. 2025. Measuring and enhancing trustworthiness of LLMs in RAG through grounded attributions and learning to refuse. In The Thirteenth International Conference on Learning Representations (ICLR 2025).

Manveer Singh Tamber, Forrest Sheng Bao, Chenyu Xu, Ge Luo, Suleman Kazi, Minseok Bae, Miaoran Li, Ofer Mendelevitch, Renyi Qu, and Jimmy Lin. 2025. Benchmarking LLM faithfulness in RAG with evolving leaderboards. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 799–811, Suzhou, China. Association for Computational Linguistics.

Vectara. 2024. HHEM-2.1-Open: Hughes hallucination evaluation model. https://huggingface.co/ vectara/hallucination\_evaluation\_model.

Jinyang Wu, Shuai Zhang, Feihu Che, Mingkuan Feng, Pengpeng Shao, and Jianhua Tao. 2025. Pandora’s box or aladdin’s lamp: A comprehensive analysis revealing the role of RAG noise in large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5019–5039, Vienna, Austria. Association for Computational Linguistics.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, et al. 2024. Qwen2.5 technical report.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380. Association for Computational Linguistics.

Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast large language model serving for RAG with cached knowledge fusion. In Proceedings of the Twentieth European Conference on Computer Systems (EuroSys ’25), pages 94–109. Association for Computing Machinery.

Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. 2025. HELMET: How to evaluate longcontext language models effectively and thoroughly.

In The Thirteenth International Conference on Learning Representations (ICLR 2025).

Tao Zhang, Ziqian Zeng, Hao Peng, Huiping Zhuang, and Cen Chen. 2025. MixKVQ: Query-aware mixedprecision KV cache quantization for long-context reasoning. Preprint, arXiv:2512.19206. Preprint.

## A Ablation Studies

## A.1 Fidelity of the Cache Round-Trip (Stage-0 Gate)

Before any quantized run, we require C1 (BF16) to exactly reproduce C0 (Oracle) on 50 held-out examples. This gate confirms that the cache roundtrip itself introduces no degradation; any difference in C2/C3 is attributable purely to quantization bitwidth. C1 reproduced C0 within floating-point tolerance on all 50 examples, validating the baseline.

## A.2 Numerical Stability

We monitored logits for NaN/Inf values across all 3,600 generations per dataset. No non-finite logits appeared under C0, C1, or C2. Under C3 (INT4), non-finite logits occurred in $< 0 . 1 \%$ of examples and were aborted rather than allowed to emit corrupted tokens. This confirms that degeneracy (Fig. 2l) reflects semantic failure, not numerical explosion.

## A.3 Faithfulness Signal Agreement

The three faithfulness signals are intentionally complementary. On RGB, HHEM and NLI entailment agree only moderately (Pearson $r { = } 0 . 4 4 )$ , confirming they capture distinct failure modes. The LLM judge achieves 92% agreement with human labels on RGB (76% on HotpotQA, where refusal artifacts add noise). Any single signal would undercount the faithfulness harm: H1 is supported by all three signals on RGB but only by the LLM judge on HotpotQA at low K, underscoring the need for a multi-signal faithfulness audit.

## A.4 Effect of Per-Block Overhead on INT4 Storage

Equation 1 predicts that INT4 with $g { = } 6 4$ achieves $\mathbf { a } \sim 3 . 6 \times$ footprint reduction over BF16 at $K { = } 5$ not the nominal $4 \times$ We verified this empirically: the per-block scale/zero-point pairs consume ≈11% of the INT4 cache footprint, shrinking the effective compression ratio from 4× to 3.6×. The INT8 footprint reduction of ${ \sim } 1 . 9 \times \ \mathrm { ( v s \ ) }$ . nominal $2 \times )$ follows the same pattern, with per-token overhead consuming ≈5% of the INT8 cache.

## A.5 H2 Power Analysis

The slope regression for H2 uses only three K values $( K ~ \in ~ \{ 1 , 3 , 5 \} )$ , leaving it statistically underpowered $\scriptstyle ( p = 0 . 1 2 )$ . Nonetheless, the directional trend is consistent: on RGB, the INT4–

BF16 hallucination gap increases from 0.09 at $K { = } 1$ to 0.26 at $K { = } 5 ;$ on HotpotQA from 0.05 at $K { = } 1$ to 0.25 at K=5. A denser sweep over $K \in \{ 1 , 2 , 3 , 4 , 5 , 7 , 1 0 \}$ would likely yield a significant slope; we leave this to future work.

Table 1: Summary: how KV-cache quantization affects accuracy and faithfulness in RAG. Each row is one observable effect; columns show what happens under each precision level and whether accuracy metrics would catch the problem on their own. ✓ = effect present / condition met; ✗ = not observed. INT8 is safe across the board; INT4 breaks faithfulness even when accuracy looks fine.
<table><tr><td>What we measured</td><td>What it means in plain INT8 terms</td><td></td><td>INT4</td><td>Does accuracy catch it?</td></tr><tr><td>Accuracy effects</td><td></td><td></td><td></td><td></td></tr><tr><td>Containment-EM drops</td><td>More answers are simply X (safe) wrong</td><td></td><td>✓(drops)</td><td>Yes, EM detects this directly</td></tr><tr><td>Token-F1 drops</td><td>Even partial matches are X (safe) worse</td><td></td><td>√ (drops)</td><td>Yes, F1 detects this directly</td></tr><tr><td colspan="3">Faithfulness effects (measured independently of accuracy)</td><td></td><td></td></tr><tr><td>HHEM hallucination rate rises</td><td>Answers are flagged as X (safe) not backed by the re- trieved documents</td><td></td><td>√ (rises)</td><td>No, EM/F1 miss this</td></tr><tr><td>NLI entailment score drops</td><td>Answers no longer logi- X (safe) cally follow from the re-</td><td></td><td>√ (drops)</td><td>No, EM/F1 miss this</td></tr><tr><td>LLM judge marks answer as unfaith- A strong LLM judges the X (safe) ful</td><td>trieved context answer as unsupported by the evidence</td><td></td><td>flips negative)</td><td>√ (&gt; 90% of No, EM/F1 miss this</td></tr><tr><td colspan="3">The &quot;hidden harm&quot;: faithfulness loss on accuracy-preserved examples (H1)</td><td></td><td>ob- √ (McNemar No, this is exactly what</td></tr><tr><td>swer string stays correct dence</td><td>Faithfulness drops even when the an- The model gives the right X (not answer but for the wrong served) reason—it is no longer using the retrieved evi-</td><td></td><td> $p { < } 1 \dot { 0 } ^ { - 2 0 }$  datasets)</td><td>, both EM/F1 cannot see</td></tr><tr><td>Does the harm grow? (H2 and H3) Harm grows with more retrieved Longer context amplifies X</td><td></td><td></td><td>√ (directional, No</td><td></td></tr><tr><td>chunks (H2)</td><td>the quantization error in the cache</td><td></td><td>K=1→5; p=0.12, under- powered)</td><td></td></tr><tr><td>Harm grows with more distractor pas-1 sages (H3, RGB only)</td><td>Noisy retrieval makes X INT4 even more likely to hallucinate</td><td></td><td>√ (β=0.22, No significant at K=3,5)</td><td></td></tr><tr><td colspan="3">Failure modes unique to INT4 Degenerate output (empty or re- The model collapses— X</td><td></td><td>√ (up to 6% Partially; EM is zero but the</td></tr><tr><td>peated text) HotpotQA refusal confound</td><td>produces nothing useful at all INT4 noise breaks the X</td><td></td><td>K=5)</td><td>of outputs at root cause is invisible √ (refusal drops No, EM and NLI are in-</td></tr><tr><td></td><td>model&#x27;s “I don&#x27;t know&quot; instinct; it guesses in- stead of refusing, inflat- ing apparent accuracy</td><td></td><td>from 51% to flated, masking harm 32% at K=1)</td><td></td></tr><tr><td colspan="3">Storage efficiency Cache size reduction vs. BF16 base- How much disk space is ~1.9× at K=5 ~3.6× at K=5 N/A</td><td></td><td></td></tr><tr><td>line (C1) saved</td><td></td><td>(vs. nominal 2×)</td><td>(vs. nominal 4×)</td><td></td></tr></table>