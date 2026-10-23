# Low-Bit N-Gram Embeddings for Language Models

Stephen Chung Wenyu Du

DualverseAI

Many recent open-weight language models use n-gram embedding tables to improve performance. These tables store learned representations of short token sequences, such as pairs or triples of consecutive tokens, which the model retrieves during processing. However, they can require substantial memory: DeepSeek-V4.1-Flash, for example, allocates 196 billion parameters to its Engram n-gram memory. We therefore investigate whether this memory cost can be reduced while preserving the performance improvement that the tables provide. We propose storing n-gram embeddings at low precision and systematically investigate diferent quantization methods and bit allocations across tables. Across nanochat-, Llama-, and GPT-2-family models trained from scratch, four-bit tables save approximately 75% of table memory while preserving more than 99% of the validation-loss improvement from adding the tables. Using fewer bits leads to a larger drop in performance, although we find that a short fine-tuning phase recovers a significant part of this loss. In addition, reinvesting the saved memory in larger tables can improve performance beyond that of the original full-precision tables while still using less memory. These table memory savings also extend to large existing models: on DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next, low-bit tables save 47–55% of table memory, with perplexity increases of less than 1.2% and largely preserved downstream task performance. These findings establish low-bit precision as a practical way to reduce the memory requirements of n-gram embeddings and use the saved capacity to improve model performance.

Date: 22-Oct-2026 Correspondence: info@dualverse.ai

DUALVERSE AI

## 1 Introduction

Recent studies have shown that n-gram embedding tables can improve language model performance with little additional computation [1, 2]. These tables store learned representations of short token sequences, such as pairs or triples of consecutive tokens. At each position, the model retrieves the corresponding representations to supplement its processing of the text. Because each lookup accesses only a few embeddings, the tables can add substantial model capacity while keeping the computation per token small. This approach has been adopted in DeepSeek’s Engram memory, which contains 196 billion parameters in DeepSeek-V4.1-Flash [3]. However, storing these large tables requires substantial memory, motivating methods that preserve their performance benefits with a smaller memory footprint.

The memory cost arises from the enormous number of possible token combinations. To accommodate them, existing methods map n-grams into a fixed number of table embeddings, allowing multiple n-grams to share the same embedding [1, 2]. Smaller tables force more unrelated sequences to share representations, which can reduce their benefit to the model. Increasing the number of embeddings alleviates this problem but requires more memory. We therefore investigate a diferent way to reduce table memory: keeping the embeddings while using fewer bits for each embedding value. This also raises an opportunity: if lower precision preserves most of the tables’ benefit, the saved memory could support more embeddings and ultimately improve performance.

We propose compressing n-gram embedding tables by storing each value at low precision. At such low precision, the choice of representable values can afect how well the embeddings are preserved. We therefore systematically compare evenly spaced values, values chosen from normal quantiles, and values fitted to the learned embedding distribution using Lloyd–Max quantization [4, 5]. We also examine whether bigram and trigram tables benefit from diferent precision levels, allowing us to allocate bits according to their sensitivity to compression. Together, these experiments ofer practical guidance on choosing quantization methods and bit allocations to balance table memory and model performance.

![](images/39655d5e603ea89c184bd2baf8e00ea9243041501bb30e54e6761f078b88b3c2.jpg)  
Figure 1: Four n-gram lookups, stored at low precision. (a) The bigram and trigram ending at the current token each retrieve two embeddings. The four embeddings are concatenated and added to the token embedding to form u<sub>t</sub>. (b) Each two-bit index selects one of four codebook values. Multiplying each selected value by the embedding’s 16-bit scale reconstructs the embedding. Codebook values use 16 bits and are shared across all embeddings of a table.

We evaluate this approach on nanochat-, Llama-, and GPT-2-family models trained from scratch [6–9]. We train each model with and without n-gram tables to establish how much the tables improve performance. We then compress the trained tables to diferent precisions and measure the table memory saved and the improvement retained. Beyond compression alone, we investigate whether a short fine-tuning phase can recover lost performance and whether reinvesting the saved memory in additional embeddings can outperform the original full-precision tables. To test whether these savings extend to large existing models, we also compress the tables in DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next, saving 47–55% of table memory while largely preserving performance on downstream tasks.

## Our main contributions are:

• Substantial table memory savings with little loss of performance. Across three model architectures, three-bit n-gram tables save 80% of table memory while retaining approximately 97% of the validation-loss improvement from adding the tables. We also demonstrate substantial table memory savings in DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next.

• Practical guidance on quantization and precision allocation. We systematically compare codebook construction, scale fitting, and precision allocation across bigram and trigram tables.

• Performance recovery through fine-tuning. A short fine-tuning phase substantially reduces the performance loss from low-bit compression, making aggressive compression more efective.

• Better performance by reinvesting saved memory. Tables with four times as many embeddings, stored at two bits, achieve 122–126% of the original tables’ improvement while still using 46–48% less table memory. This shows that reducing precision can enable larger, more efective n-gram memories.

## 2 Method

N-gram embedding tables. Following the Over-Tokenized Transformer [1], we use four embedding tables: two for bigrams and two for trigrams, as illustrated in Figure 1. The embeddings in these tables are learned jointly with the language model, with one embedding stored in each row. At each token position, the bigram and trigram ending at the current token each retrieve one embedding from each of their two corresponding tables. These four embeddings are then concatenated and added to the token embedding. Previous work has shown that this method improves language modeling performance, with further gains as the number of embeddings increases [1].

Formally, let $x _ { t }$ denote the current token, $E$ the token embedding matrix with embedding dimension $d ,$ and $T _ { 1 } , \ldots , T _ { 4 }$ the four tables. We construct the model input as

$$
u _ { t } = E [ x _ { t } ] + \mathrm { C o n c a t } _ { j = 1 } ^ { 4 } T _ { j } \left[ h _ { j } ( x _ { t - n _ { j } + 1 } , \dots , x _ { t } ) \right] ,\tag{1}
$$

where $n _ { 1 } = n _ { 2 } = 2$ for the bigram tables and $n _ { 3 } = n _ { 4 } = 3$ for the trigram tables. Here, the hash function $h _ { j }$ determines which embedding is retrieved by mapping the token sequence to an address in table $T _ { j }$ . These retrieved embeddings each have dimension $d / 4 ,$ , so concatenating all four gives the same dimension as $E [ x _ { t } ]$

Low-bit embeddings. We propose reducing the memory cost of these trained tables by storing their embed dings at low precision. To do this, we give each table a set of up to $2 ^ { b }$ representative values, commonly called a codebook, and represent each embedding component by a b-bit index into these shared codebook values. To account for diferences in magnitude between embeddings, we also store one 16-bit scale per embedding. During reconstruction, the index selects a codebook value, which is then multiplied by the embedding’s scale. Thus, the original component $w _ { r i }$ at position i of embedding r is reconstructed as

$$
\hat { w } _ { r i } = s _ { r } c _ { q _ { r i } } , \qquad q _ { r i } \in \{ 0 , \ldots , 2 ^ { b } - 1 \} ,\tag{2}
$$

where $s _ { r }$ is the scale, $c _ { k }$ is codebook value $k ,$ and $q _ { r i }$ is the stored index (Figure 1b).

This low-bit representation substantially reduces table memory. For an embedding with $p$ components, memory use falls from 16p bits to $p b + 1 6$ bits. For instance, with $p = 1 2 8 .$ , two-bit precision reduces the memory required per embedding by approximately 87%, from 2,048 to 272 bits.

However, bigram and trigram embeddings represent diferent context patterns, so they may need diferent precision to preserve their contribution to model performance. We therefore propose using a mixture of bit widths across the two n-gram orders, allocating more bits to the tables that are more sensitive to quantization within a fixed table memory budget.

Choosing the codebook values. We consider three codebook families. First, uniform codebooks use equally spaced values. For instance, at two bits, the codebook values are $\{ - 1 , - 1 / 3 , 1 / 3 , 1 \}$ . We also test a variant that includes zero, using $\{ - 1 , 0 , 1 \}$ , so that zero can be represented exactly. Second, normal-quantile codebooks take values from quantiles of a standard normal distribution, placing more of them near zero. Third, we learn the codebook values from each table’s trained embeddings using Lloyd–Max quantization [4, 5].

To learn the Lloyd–Max codebook, we first normalize each embedding by its root mean square, allowing embeddings of diferent magnitudes to share a codebook. We initialize the codebook values at quantiles of the normalized components, then repeatedly assign components to their nearest codebook value and replace each value with the mean of its assigned components. This fitting runs once during compression, using only the trained embeddings. Appendix $\mathrm { A . 2 }$ gives the implementation details.

Choosing the embedding scales. Once the codebook values are fixed, we consider three ways to choose each embedding’s scale $s _ { r } .$ . First, maximum-based scaling matches the largest absolute codebook value to the embedding’s largest absolute component. Second, root-mean-square (RMS) scaling uses the embedding’s overall magnitude, calculated as $s _ { r } = \sqrt { \frac { 1 } { p } \sum _ { i } w _ { r i } ^ { 2 } }$ . Third, error-fitted scaling chooses the scale to minimize the mean squared error between the original embedding and its reconstruction. We compare these choices to examine how scale selection afects compression quality.

After determining the codebook values and embedding scales, we store them together with the low-bit indices in place of the original embedding tables. During inference, the model retrieves embeddings from this stored representation without repeating the fitting procedure.

## 3 Experiments

## 3.1 Experimental setup

Models and data. We train five language models from scratch: nanochat with 26M and 50M parameters, a GPT-2-family model with 30M parameters, and Llama-family models with 31M and 322M parameters. By default, each model uses four n-gram tables, comprising two bigram and two trigram tables, with approximately 262k embeddings per table. We train on ClimbMix [10] and repeat each experiment with three random seeds. Appendix B.1 gives the model configurations and training settings.

Evaluation. For each model, we first train two versions using the same experimental setup, one with fullprecision (16-bit) n-gram tables and one without tables. We evaluate both on a held-out validation dataset, and their diference in validation loss measures the improvement from adding the tables. We then reduce the precision of the trained tables and evaluate the model again to measure how much of this improvement remains. We report this as the percentage of the original improvement retained, where 100% matches the full-precision tables and 0% matches the model trained without tables. Alongside this performance measure, we report the percentage of table memory saved. Results are averaged over three seeds, with standard deviations where shown.

## 3.2 Table memory savings and model performance

We first ask how much table memory can be removed before the model loses the benefit of n-gram embeddings. We compress the trained tables to one through four bits using Lloyd–Max codebooks with fitted scales. To test whether larger tables can tolerate the same compression, we also increase the number of embeddings per table from the default 262k to 1M in the 26M nanochat model and both Llama-family models.

The results are shown in Figure 2. We find that low-bit compression retains most of the improvement at three or four bits. For example, four-bit tables save 74–75% of table memory while retaining more than 99% of the original improvement. However, reducing precision further hurts performance. Two-bit tables save 86–87% of table memory but retain only 85.9–91.0% of the original improvement. Appendix C gives the full results, including one-bit compression.

## 3.3 Codebook choices and mixed precision

Codebook values and embedding scales. To understand how best to choose codebook values and embedding scales, we compare the three codebook families and scale choices described in Section 2. We compare these choices using two-bit tables in the 31M Llama-family model.

The results are shown in Figure 3a. We observe that choosing an appropriate embedding scale matters more than the choice of codebook. With error-fitted scales, the three codebooks retain a similar share of the tables’ improvement (86.36–86.52%), whereas uniform codebooks with maximum-based scaling retain only 58.04%. Lloyd–Max also performs well with simple RMS scaling, retaining 86.19%. Among the three fitted codebooks, Lloyd–Max provides a small additional gain, so we use it as the default in the remaining experiments.

Mixed precision. Since bigram and trigram tables may require diferent precision, we also test all nine combinations of two-, three-, and four-bit precision. Combinations with the same total bit count have approximately equal memory cost, allowing us to compare how precision should be distributed between bigram and trigram tables. Giving more bits to bigrams consistently retains more of the improvement than the reverse allocation (Figure 3b). In particular, four-bit bigrams and three-bit trigrams retain 98.9–99.4% of the improvement while saving 77–78% of table memory across the five models. Appendix C gives the full results for all precision combinations.

![](images/2fc3f5a184d3503e7f3f7f2aa89c899436be7df62a473a31d7717f76780ef687.jpg)  
Figure 2: Four-bit embeddings save about 75% of table memory while retaining more than 99% of the improvement. (a) The five models with the default 262k embeddings per table. (b) The 26M nanochat model and both Llama-family models with 1M embeddings per table. Points and error bars show means and sample standard deviations over three seeds. The dashed line matches the original 16-bit tables.

(a) Codebook and scale  
![](images/76ccd2e3e3109adc87fa86ecccf8c219f05c33b846dee72252a94666dfa77505.jpg)

(b) Mixed precision  
![](images/219e89731331b8bac02041c244cc9e9c6ae6f4fcf2ab1974823d423cac504208.jpg)  
Figure 3: Choosing an appropriate embedding scale is more important than choosing among uniform, normalquantile, and Lloyd–Max codebooks. (a) Two-bit comparisons on the 31M Llama-family model with 262k embeddings per table. Uniform\* is the three-value variant {−1, 0, 1} that includes zero. Error bars show sample standard deviations over three seeds. (b) Selected bigram/trigram precision pairs for all five models with the default 262k embeddings per table. Appendix C.3 gives all precision pairs.

## 3.4 Evaluation on large existing models

We next test whether low-bit compression can reduce the memory used by n-gram embedding tables in large open-weight language models. We evaluate DeepSeek-V4.1-Flash [3] and Qwen3.8-Flash-Next [11], compressing their n-gram tables while keeping the rest of each model unchanged.

Uniform precision. We first test whether using one low precision for all n-gram embedding tables can save substantial table memory without noticeably changing model performance. We compare two- to five-bit tables with the original tables using perplexity on WikiText-2 validation data [12]. The results are shown in Figure 4a. Four-bit tables save about half of the table memory, with a 1.26% perplexity increase for DeepSeek and almost unchanged perplexity for Qwen. Stronger compression afects the models diferently: at three bits, Qwen’s perplexity increases by only 0.52%, compared with 5.06% for DeepSeek.

(a) Uniform precision  
![](images/bfe99b3a155751a6daeb7d2ac4006233e6fb5741673d271064285941b6b97968.jpg)

(b) Mixed precision  
![](images/4f4973afc95cfb08d91e1e326d15e854a07686f465574bf1f6847509ecd554f6.jpg)  
Figure 4: Uniform and mixed precision provide diferent memory–performance choices in DeepSeek-V4.1- Flash and Qwen3.8-Flash-Next. (a) Two through five bits for every table, with labels giving bits per value. (b) Representative mixed allocations, ordered by n-gram length. Both panels use WikiText-2 validation perplexity.

Mixed precision. To see whether some n-gram tables need more precision than others, we test all combinations of two through five bits across the n-gram orders. Figure 4b shows that the precision assignment matters even at the same memory cost. For DeepSeek, giving an extra bit to trigrams or four-grams $( 4 / 5 / 4$ or $4 / 4 / 5 )$ preserves perplexity better than giving it to bigrams $( 5 / 4 / 4 )$ . In Qwen, the extra bit is more useful for bigrams: 4/3 performs better than 3/4. The better assignment therefore difers between the two models.

Besides WikiText-2, we further evaluate DeepSeek with 4/5/4-bit tables and Qwen with 4/3-bit tables on Penn Treebank and C4 [13, 14]. Across all three held-out datasets, these settings keep perplexity changes to less than 1.2% while saving 47–55% of table memory.

Downstream performance. In addition to perplexity, which measures next-token prediction, we examine how low-bit tables afect downstream task performance. We evaluate DeepSeek with 4/5/4-bit tables and Qwen with 4/3-bit tables on HellaSwag, ARC-Challenge, and WinoGrande [15–17]. Average accuracy across these tasks decreases by only 0.93% for DeepSeek and 0.27% for Qwen (Figure 5). These results show that the table memory savings also come with little change in downstream performance.

![](images/28dca220464792123d2712acd1930c409d444cafdc40be3d0d0346c508acf385.jpg)  
Figure 5: Downstream evaluation of the selected mixed-precision settings. Change in accuracy (%) for DeepSeek 4/5/4 and Qwen 4/3, computed as low-bit accuracy minus original accuracy. Error bars show paired 95% bootstrap confidence intervals. The settings save 46.7% and 55.0% of table memory, respectively.

## 3.5 Recovering performance with low-bit training

Compressing the n-gram tables changes the embeddings that the model learned to use, which can reduce performance. We therefore ask whether a short fine-tuning phase can recover this loss, or whether using low-bit embeddings throughout pre-training is more efective. To train with low-bit embeddings, we maintain a BF16 copy for gradient updates but quantize the retrieved embeddings during each forward pass. Both the embeddings and the rest of the model remain trainable, allowing the model to adapt to the chosen precision. We compare four sequences: (1) compress → pre-train, (2) pre-train → compress → fine-tune, (3) pre-train → fine-tune → compress, and (4) pre-train → compress. Comparing fine-tuning before and after compression tests whether recovery comes from adapting to low-bit embeddings rather than simply receiving more training.

The results are summarized in Figure 6. We observe that a brief fine-tuning phase after compression recovers much of the performance lost at very low precision. After fine-tuning, one- and two-bit tables retain 65.3% and 92.0% of the improvement provided by full-precision tables, respectively. Surprisingly, one-bit embeddings perform even better when used throughout pre-training, retaining 77.0% of the full-precision tables’ improvement. This shows that even one bit per embedding component can preserve a large share of the benefit from n-gram tables when the model learns to use them from the start. Appendix C.5 gives the full results, while Appendix C.6 examines how alternative hashing methods afect performance with low-bit embeddings.

## 3.6 Reinvesting memory in larger tables

Finally, we ask whether the memory saved by compression can support larger tables and improve performance. We use tables with 262k embeddings stored at 16-bit precision as the baseline. We then increase each table to 1M embeddings and test lower precisions to see whether the larger tables can improve performance while using less memory than the baseline. We measure performance relative to the improvement that the baseline tables provide over the model without n-gram embeddings.

The results are shown in Figure 7. We observe that reinvesting the memory saved by compression in more embeddings can improve performance while still reducing total table memory. For example, increasing each table from 262k to 1M embeddings at two-bit precision uses only 52–54% of the original table memory while providing 22–26% more improvement than the original 16-bit tables. Three-bit tables improve performance further while using 77–79% of the original table memory.

We also test a brief fine-tuning phase on the compressed larger tables, following Section 3.5, to assess whether adaptation brings further improvement. Fine-tuning improves performance across all three models, with the largest recovery at one bit: with the same fine-tuning budget, compressing before fine-tuning achieves 97– 111% of the original tables’ improvement, compared with 44–63% when compressing after fine-tuning. At two bits, fine-tuning raises the improvement from 122–126% to 129–145% at the same memory cost. These results suggest a practical approach to memory-eficient deployment: pre-train with more embeddings, then compress the tables and briefly fine-tune the model to recover performance.

![](images/3e30ba2e100447add0d6edd6e8048fbd3aaa05b1052ab9ed376d15535ec0b5be.jpg)

Figure 6: Fine-tuning after compression recovers much of the n-gram improvement. Improvement retained on the 31M Llama-family model, relative to full-precision and no-table models with the same training budget. Error bars show sample standard deviations over three seeds  
(a) nanochat 26M  
![](images/7452dc54f60c41cb1cdd47597e8338a34d42ca72945038fd64893506a0e4733f.jpg)

(b) Llama-family 31M  
(c) Llama-family 322M  
![](images/3682ddf624a5076a9e9439e0f0c35eb1355e0d8817710efebf1697675fec1e8f.jpg)  
Figure 7: Larger low-bit tables can improve performance while using less memory. Both axes use the original 262k-embedding, 16-bit tables as the reference, marked at (100%, 100%). The shaded region indicates greater improvement while using less memory. Fine-tuning follows compression. Error bars show sample standard deviations over three seeds.

## 4 Related work

N-gram and lookup memory in language models. Learned local-context representations go back to neural n-gram models [18], and hashing shares parameters among many keys [19, 20]. N-Grammer adds n-gram embeddings over discrete latent codes [21], and BLT uses hashed n-gram embeddings over bytes [22]. The Over-Tokenized Transformer [1] hashes multi-gram inputs into large input embedding tables, the operation we study. Deep Sparse Embedding (DSE) [2] places hashed n-gram memory inside the network with contextaware gating and short convolutions, balances it against mixture-of-experts capacity, and serves it from host memory. Other designs factorize n-gram embeddings [23, 24] or use frequency-aware hashing and alias mixing [25]. SCONE precomputes contextualized n-gram embeddings and stores them of-accelerator [26]. Memory layers retrieve from a product-key memory [27, 28], and external datastores add capacity at inference time [29, 30].

Compressing embedding tables. Large embedding tables in recommendation systems are compressed by perembedding four-bit quantization with codebooks [31], compositional embeddings [32], and mixed embedding dimensions [33]; word embeddings can be replaced by learned compositional codes [34]. Because hashed embeddings are already shared, we compare fewer embeddings with fewer bits per embedding directly.

## 5 Conclusion

We propose low-bit precision for n-gram embedding tables and systematically study its efects on memory use and model performance. Across models trained from scratch, four-bit tables save approximately 75% of table memory while preserving more than 99% of the improvement provided by full-precision tables. Choosing appropriate embedding scales matters more than the choice of codebook, and mixed precision allows diferent n-gram tables to use the precision they need. At lower precision, a brief fine-tuning phase can recover much of the lost performance. These findings also extend to DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next, where compression saves 47–55% of table memory with little change in perplexity and downstream performance. Beyond reducing memory use, reinvesting the savings in more embeddings can improve performance over the original full-precision tables. Together, our results ofer practical guidance for making n-gram memory more eficient and using the saved capacity to build better language models.

## 6 AI Use

This work builds on a finding made by an autonomous agent run by the authors on rStation, a multi-agent research platform. Given the open-ended task of improving language-model pre-training, the agent found that n-gram embedding tables could be stored at low precision with minimal performance loss. The agent published its findings at https://rstation.ai/problems/general\_pretraining/papers/6/v4.

The authors considered this finding potentially useful to the broader research community and therefore reproduced and extended the experiments outside rStation, using additional models and datasets. The authors also revised the manuscript to make the findings accessible to researchers.

## References

[1] Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, and Xun Zhou. Overtokenized transformer: Vocabulary is generally worth scaling. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 26261–26282, 2025.

[2] Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Yukun Li, Han Zhang, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. Conditional memory via scalable lookup: A new axis of sparsity for large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4968–4990, 2026. URL https://aclanthology.org/2026.acl-long.226/.

[3] DeepSeek-AI. DeepSeek-V4.1-Flash: Model card. https://huggingface.co/deepseek-ai/ DeepSeek-V4.1-Flash, 2026. Accessed October 19, 2026.

[4] Stuart P. Lloyd. Least squares quantization in PCM. IEEE Transactions on Information Theory, 28 (2):129–137, 1982.

[5] Joel Max. Quantizing for minimum distortion. IRE Transactions on Information Theory, 6(1):7–12, 1960.

[6] Andrej Karpathy. nanochat. https://github.com/karpathy/nanochat, 2025.

[7] Andrej Karpathy. autoresearch. https://github.com/karpathy/autoresearch, commit 228791f, 2026.

[8] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[9] Alec Radford, Jefrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019.

[10] Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, Mostofa Patwary, Yingyan Lin, Jan Kautz, and Pavlo Molchanov. Nemotron-CLIMB: Clustering-based iterative data mixture bootstrapping for language model pre-training. In Advances in Neural Information Processing Systems, volume 38, 2025.

[11] Qwen Team. Qwen3.8-Flash-Next: Model card. https://huggingface.co/Qwen/Qwen3. 8-Flash-Next, 2026. Accessed October 20, 2026.

[12] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017.

[13] Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The Penn Treebank. Computational Linguistics, 19(2):313–330, 1993.

[14] Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

[15] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, 2019.

[16] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge, 2018.

[17] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020.

[18] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155, 2003.

[19] Wenlin Chen, James Wilson, Stephen Tyree, Kilian Weinberger, and Yixin Chen. Compressing neural networks with the hashing trick. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, 2015.

[20] Dan Tito Svenstrup, Jonas Hansen, and Ole Winther. Hash embeddings for eficient word representations. In Advances in Neural Information Processing Systems, volume 30, 2017.

[21] Aurko Roy, Rohan Anil, Guangda Lai, Benjamin Lee, Jefrey Zhao, Shuyuan Zhang, Shibo Wang, Ye Zhang, Shen Wu, Rigel Swavely, et al. N-Grammer: Augmenting transformers with latent n-grams. arXiv preprint arXiv:2207.06366, 2022.

[22] Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, and Srinivasan Iyer. Byte latent transformer: Patches scale better than tokens. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pages 9238–9258, 2025.

[23] Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, Yuning Qiu, Qibin Zhao, and Danilo Mandic. Tensorizing engram: Sharing latents across n-gram embeddings is beneficial in LLMs. arXiv preprint arXiv:2606.08347, 2026.

[24] Bowen Yang, Jingbo Zhou, Qinghong Miao, and Hua Wu. FactorEngram: Factorized n-gram memory with basis-level gating for language models. arXiv preprint arXiv:2609.35578, 2026.

[25] Yilong Chen, Yanxi Xie, Zitian Gao, He Xin, Yihao Xiao, Jason Klein Liu, Haoming Luo, Yifan Luo, Zhengmao Ye, Tingwen Liu, Xin Zhao, Ran Tao, and Bryan Dai. Beyond n-gram: Data-aware X-GRAM extraction for eficient embedding parameter scaling. arXiv preprint arXiv:2604.21724, 2026.

[26] Da Yu, Edith Cohen, Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Daogao Liu, and Chiyuan Zhang. Scaling embedding layers in language models. In Advances in Neural Information Processing Systems, volume 38, pages 31969–31995, 2025.

[27] Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jé- gou. Large memory layers with product keys. In Advances in Neural Information Processing Systems, volume 32, 2019.

[28] Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, and Gargi Ghosh. Memory layers at scale. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 3831–3842, 2025.

[29] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations, 2020.

[30] Sebastian Borgeaud, Arthur Mensch, Jordan Hofmann, Trevor Cai, Eliza Rutherford, Katie Millican, et al. Improving language models by retrieving from trillions of tokens. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 2022.

[31] Hui Guan, Andrey Malevich, Jiyan Yang, Jongsoo Park, and Hector Yuen. Post-training 4-bit quantization on embedding tables. In MLSys Workshop at Advances in Neural Information Processing Systems, 2019.

[32] Hao-Jun Michael Shi, Dheevatsa Mudigere, Maxim Naumov, and Jiyan Yang. Compositional embeddings using complementary partitions for memory-eficient recommendation systems. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 165–175, 2020.

[33] Antonio A. Ginart, Maxim Naumov, Dheevatsa Mudigere, Jiyan Yang, and James Zou. Mixed dimension embeddings with application to memory-eficient recommendation systems. In IEEE International Symposium on Information Theory, pages 2786–2791, 2021.

[34] Raphael Shu and Hideki Nakayama. Compressing word embeddings via deep compositional code learning. In International Conference on Learning Representations, 2018.

[35] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. SmolLM2: When smol goes big – data-centric training of a fully open small language model. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id= 3JiCl2A14H.

## A Detailed methods

## A.1 N-gram hashing and lookup

Following the Over-Tokenized Transformer [1], we retrieve embeddings by hashing the bigram and trigram ending at each token. For the token ID $x _ { t }$ at position $t ,$ vocabulary size $V = 8 1 9 2$ , and base $B = V + 1$ , our from-scratch experiments use the keys

$$
k _ { t } ^ { ( 2 ) } = x _ { t } + B x _ { t - 1 } , \qquad k _ { t } ^ { ( 3 ) } = x _ { t } + B x _ { t - 1 } + B ^ { 2 } x _ { t - 2 } .\tag{3}
$$

Missing predecessors take the value V. The bigram addresses are $k _ { t } ^ { ( 2 ) }$ mod m and $k _ { t } ^ { ( 2 ) }$ mod $( m + 2 )$ , and the trigram addresses are $k _ { t } ^ { ( 3 ) }$ mod $( m + 4 )$ and $k _ { t } ^ { ( 3 ) }$ mod $( m + 6 )$ . The base table size is $m = 2 6 2 , 1 4 1$ by default and $m = 1 , 0 4 8 , 5 7 3$ for the 1M-embedding experiments. The retrieved embeddings are then used as described in Section 2.

## A.2 Codebook construction

As described in Section 2, we consider three codebook families: uniform, normal-quantile, and Lloyd–Max. Each codebook supplies the representative values shared by the embeddings in one table. With b bits per component, a codebook can contain up to $K = 2 ^ { b }$ values, denoted $c _ { 0 } , \ldots , c _ { K - 1 }$

Uniform codebooks. We place K values at equal intervals between −1 and 1. At two bits, this gives $\{ - 1 , - 1 / 3 , 1 / 3 , 1 \}$ . We also test the zero-centered variant, denoted Uniform\*, which uses $K - 1$ equally spaced values so that zero is included. Its two-bit codebook is $\{ - 1 , 0 , 1 \}$

Normal-quantile codebooks. We place values at quantiles of a standard normal distribution, concentrating more values near zero. For more than two bits, we take quantiles at probabilities $( k + 1 / 2 ) / K$ for $k =$ $0 , \ldots , K - 1$ , then divide all values by their largest absolute value. At two bits, we instead use the fixed codebook $\{ - 1 , - 0 . 3 1 8 6 , 0 . 3 1 8 6 , 1 \}$

Lloyd–Max codebooks. To adapt the codebook to the trained embeddings, we learn its values from their component distribution [4, 5]. We first normalize each embedding so that diferences in magnitude do not dominate the shared codebook. Let $w _ { r i }$ be component i of embedding $r ,$ and let $p$ be its number of components. The initial normalization scale $s _ { r } ^ { ( 0 ) }$ is the embedding’s root mean square, and $z _ { r i }$ is its normalized component:

$$
s _ { r } ^ { ( 0 ) } = \sqrt { \frac { 1 } { p } \sum _ { i = 1 } ^ { p } w _ { r i } ^ { 2 } } , \qquad z _ { r i } = \frac { w _ { r i } } { s _ { r } ^ { ( 0 ) } } .\tag{4}
$$

In implementation, we clamp the normalization scale to at least $1 0 ^ { - 1 2 }$ to avoid division by zero.

We then fit the codebook to these normalized components. We sample at most 4M components per table and initialize the K codebook values at quantiles $( k + 1 / 2 ) / K$ of this sample. Each Lloyd–Max iteration has two steps. First, we assign every sampled component to its nearest codebook value. Second, we replace each codebook value with the mean of the components assigned to it. Formally, for sampled components indexed by j, the assignment $a _ { j }$ and updated value $c _ { k } ^ { \mathrm { n e w } }$ are

$$
a _ { j } = \underset { 0 \leq k < K } { \arg \operatorname* { m i n } } | z _ { j } - c _ { k } | ^ { 2 } , \qquad c _ { k } ^ { \mathrm { n e w } } = \frac { 1 } { | S _ { k } | } \sum _ { j \in S _ { k } } z _ { j } , \quad S _ { k } = \{ j : a _ { j } = k \} .\tag{5}
$$

Here, $\scriptstyle { S _ { k } }$ contains the indices of the sampled components assigned to value k. If this set is empty, that codebook value is unchanged. We repeat the two steps 60 times and store the fitted values in bfloat16 (BF16). Once the codebook is fixed, we choose the embedding scales using one of the methods below.

## A.3 Embedding scales

As described in Section 2, we compare three methods for choosing embedding scales: maximum-based, root-mean-square (RMS), and error-fitted scaling. For embedding $r ,$ the scale $s _ { r }$ multiplies every selected codebook value, so component $w _ { r i }$ is reconstructed as $s _ { r } c _ { q _ { r i } }$ , where $q _ { r i }$ is the index of the selected codebook value. The methods below difer in how they choose $s _ { r }$ for a fixed codebook.

Maximum-based scaling. We match the largest absolute codebook value to the embedding’s largest absolute component. With codebook values $c _ { k } .$ , the scale is

$$
s _ { r } ^ { \mathrm { m a x } } = \frac { \operatorname* { m a x } _ { i } | w _ { r i } | } { \operatorname* { m a x } _ { k } | c _ { k } | } .\tag{6}
$$

Root-mean-square (RMS) scaling. We use the embedding’s overall magnitude as the scale, i.e.

$$
s _ { r } ^ { \mathrm { R M S } } = \sqrt { \frac { 1 } { p } \sum _ { i = 1 } ^ { p } w _ { r i } ^ { 2 } } .\tag{7}
$$

Error-fitted scaling. To choose a scale based on reconstruction quality, we test the RMS scale and a range of scales derived from maximum-based scaling. Specifically, the candidates are $s _ { r } ^ { \mathrm { R M S } }$ and $\alpha s _ { r } ^ { \mathrm { m a x } }$ , where α is a multiplier taking values 1.00, 0.95, . . . , 0.30. A smaller α brings the representative values closer together, allowing finer approximation of smaller components while clipping larger ones. For each candidate, we round the scale to BF16, assign each component to its nearest scaled codebook value, and reconstruct the embedding in BF16. We select the candidate with the lowest mean squared error relative to the original embedding. This is the error-fitted method used in the main compression comparisons.

## B Experimental setup

## B.1 Models and training settings

We train five models from scratch to test compression across architectures and model sizes. Table 1 lists their dimensions and n-gram table sizes. The two nanochat models follow Karpathy [6, 7], while the other models use GPT-2-family [9] and Llama-family architectures, with the latter adapted from SmolLM2 [35]. We refer to these implementations for the remaining architectural details. All models use an 8,192-token vocabulary and the same n-gram lookup design. The source code underlying the experiments can be found at https://github.com/dualverse-ai/lowbit-ngram.

Table 1: Model dimensions and n-gram table sizes. Parameter counts exclude the n-gram tables, and d denotes model width.
<table><tr><td>Model</td><td>Parameters (millions)</td><td>Layers</td><td>Width d</td><td>Embeddings per table</td></tr><tr><td>nanochat 26M</td><td>26.35</td><td>6</td><td>384</td><td>262k, 1M</td></tr><tr><td>nanochat 50M</td><td>50.33</td><td>8</td><td>512</td><td>262k</td></tr><tr><td>GPT-2-family 30M</td><td>30.46</td><td>8</td><td>512</td><td>262k</td></tr><tr><td>Llama-family 31M</td><td>30.94</td><td>8</td><td>512</td><td>262k, 1M</td></tr><tr><td>Llama-family 322M</td><td>322.50</td><td>32</td><td>960</td><td>262k, 1M</td></tr></table>

Table 2 lists the optimizers, peak learning rates, and training budgets. Models with and without tables use the same training setup, and the table embeddings are initialized to zero.

Table 2: Training settings. For nanochat, the backbone learning rate applies to Muon-updated matrices. Other parameter groups follow the original implementation.
<table><tr><td>Setting</td><td>nanochat</td><td>GPT-2 30M / Llama 31M</td><td>Llama 322M</td></tr><tr><td colspan="4">Optimization</td></tr><tr><td>Backbone optimizer</td><td>Muon / AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Table optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Backbone learning rate</td><td>0.04</td><td>0.001</td><td>0.0006</td></tr><tr><td>Table learning rate</td><td>0.6√768/d</td><td>0.003</td><td>0.0018</td></tr><tr><td colspan="4">Training budget</td></tr><tr><td>Updates</td><td>953</td><td>7,624</td><td>7,630</td></tr><tr><td>Target tokens per update</td><td>524,288</td><td>65,536</td><td>131,072</td></tr></table>

## B.2 Training and validation data

Training uses the shufled ClimbMix release [10], tokenized with an 8,192-token byte-pair encoding tokenizer. We pack documents into sequences of 2,049 tokens, providing 2,048 prediction targets per sequence. The small models train on approximately 500M prediction targets, and the 322M model trains on approximately 1B prediction targets.

We use separate validation datasets to measure final performance. These datasets come from held-out portions of ClimbMix. We remove duplicate validation documents and any documents also present in training or development. The validation set contains 10,240 packed sequences, totaling 20.97M prediction targets. All compared models are scored on the same sequences.

## B.3 DeepSeek and Qwen evaluation

For DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next, we compress the n-gram tables while keeping the rest of each model fixed. Their original tables are stored at eight-bit floating-point (FP8) precision, which we use as the baseline when measuring table memory savings. We use WikiText-2 validation data to compare precision allocations. DeepSeek has bigram, trigram, and four-gram tables, while Qwen has bigram and trigram tables. For both models, we use Lloyd–Max codebooks with error-fitted embedding scales as the default for evaluation.

To test whether promising allocations transfer beyond the selection data, we evaluate them on held-out WikiText-2, Penn Treebank, and C4. Each corpus contributes 64 windows and 32,768 scored target tokens, with identical windows for the original and compressed models.

We evaluate the models on HellaSwag, ARC-Challenge, and WinoGrande. HellaSwag and ARC-Challenge use length-normalized accuracy, while WinoGrande uses accuracy. We evaluate their full splits of 10,042, 1,172, and 1,267 examples, respectively. Accuracy diferences compare predictions on the same examples, with 95% confidence intervals from 100,000 paired bootstrap samples.

## B.4 Training and fine-tuning with low-bit embeddings

This section describes the training and fine-tuning procedures used in Sections 3.5 and 3.6.

Training with low-bit embeddings. We retain a BF16 copy of each embedding for gradient updates and quantize the retrieved embeddings in the forward pass. During backpropagation, we treat quantization as an identity operation, passing gradients to the BF16 copy through a straight-through estimator. Embedding scales are fitted on each forward pass, and the Lloyd–Max codebook is refitted every update for the first 100 updates and every 25 updates thereafter.

Fine-tuning details. For the 262k-embedding tables, fine-tuning uses 381 additional updates on unseen training sequences, with 32 sequences per update. All weights remain trainable, using fresh AdamW states, 20 warmup updates, and a learning rate of 0.1 times the original peak.

For the 1M-embedding experiments, fine-tuning uses new training sequences for the smaller models and reuses pre-training sequences for the 322M model.

## B.5 Computational resources

The from-scratch experiments run on RTX 5090 GPUs. Small-model training runs take approximately 10–25 minutes on one GPU, while a 322M run takes 6.4–6.8 hours. DeepSeek and Qwen evaluations use four H200 GPUs with tensor parallelism of four.

Table memory. We compute table memory savings from the memory required by the n-gram embedding tables. The compressed table memory includes the low-bit embedding indices, one 16-bit scale per embedding, and the shared codebook values. We report the percentage reduction relative to the original tables, stored at 16 bits in our from-scratch experiments and eight bits in DeepSeek and Qwen. These percentages describe table memory rather than total model memory.

## C Complete experimental results

## C.1 Compression and table size

Here we present the full compression results for Section 3.2 (Figure 2) and the table-size comparison in Section 3.6 (Figure 7). Tables 3 and 4 report results for 262k and 1M embeddings per table, respectively. Both use each model’s 262k-embedding, 16-bit tables as the reference for computing Loss Change and Improvement.

Table 3: Compression with 262k embeddings per table, using Lloyd–Max codebooks and error-fitted scales. Validation loss, loss change, and improvement are means ± sample standard deviations over three seeds.
<table><tr><td>Precision (bits)</td><td>Memory used (%)</td><td>Validation loss (bits/byte)</td><td>Loss change (%)</td><td>Improvement (%)</td><td></td></tr><tr><td colspan="6">nanochat 26M</td></tr><tr><td>1</td><td>7.3</td><td> $1 . 0 3 9 6 3 \pm 0 . 0 0 0 3 2$ </td><td> $2 . 1 3 2 \pm 0 . 0 4 9$ </td><td> $2 9 . 0 \pm \ : 2 . 8$ </td><td></td></tr><tr><td>2</td><td>13.5</td><td> $1 . 0 2 1 6 4 \pm 0 . 0 0 0 6 6$ </td><td> $0 . 3 6 5 \pm 0 . 0 0 4$ </td><td> $8 7 . 9 \pm 0 . 6$ </td><td></td></tr><tr><td>3</td><td>19.8</td><td> $1 . 0 1 8 7 6 \pm 0 . 0 0 0 7 0$ </td><td> $0 . 0 8 2 \pm 0 . 0 0 1$ </td><td> $9 7 . 3 \pm . 0 . 2$ </td><td></td></tr><tr><td>4</td><td>26.0</td><td> $1 . 0 1 8 1 2 \pm 0 . 0 0 0 7 0$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 0$ </td><td> $9 9 . 4 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td>16</td><td>100.0</td><td> $1 . 0 1 7 9 3 \pm 0 . 0 0 0 7 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 0 0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td>No table</td><td>0.0</td><td> $1 . 0 4 8 5 5 \pm 0 . 0 0 1 3 9$ </td><td> $3 . 0 0 8 \pm 0 . 1 7 8$ </td><td> $0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td colspan="6">nanochat 50M</td></tr><tr><td>1</td><td>7.0</td><td> $0 . 9 8 9 3 5 \pm 0 . 0 0 0 9 3$ </td><td> $1 . 6 3 6 \pm 0 . 0 0 6$ </td><td> $2 2 . 7 \pm \ : 2 . 4$ </td><td></td></tr><tr><td>2</td><td>13.3</td><td> $0 . 9 7 6 0 6 \pm 0 . 0 0 0 8 7$ </td><td> $0 . 2 7 2 \pm 0 . 0 0 2$ </td><td> $8 7 . 2 \pm 0 . 4$ </td><td></td></tr><tr><td>3</td><td>19.5</td><td> $0 . 9 7 4 0 2 \pm 0 . 0 0 0 8 6$ </td><td> $0 . 0 6 1 \pm 0 . 0 0 1$ </td><td> $9 7 . 1 \pm \ : \ : 0 . 1$ </td><td></td></tr><tr><td>4</td><td>25.8</td><td> $0 . 9 7 3 5 6 \pm 0 . 0 0 0 8 6$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 0$ </td><td> $9 9 . 3 \pm 0 . 0$ </td><td></td></tr><tr><td>16</td><td>100.0</td><td> $0 . 9 7 3 4 2 \pm 0 . 0 0 0 8 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 0 0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td>No table</td><td>0.0</td><td> $0 . 9 9 4 0 3 \pm 0 . 0 0 1 3 1$ </td><td> $2 . 1 1 7 \pm 0 . 0 7 0$ </td><td> $0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td colspan="6">GPT-2-family 30M</td></tr><tr><td>1</td><td>7.0</td><td> $1 . 1 2 3 6 2 \pm 0 . 0 0 4 4 1$ </td><td> $4 . 3 5 6 \pm 0 . 0 8 6$ </td><td> $1 6 . 9 \pm 3 3 . 8$ </td><td></td></tr><tr><td>2</td><td>13.3</td><td> $1 . 0 8 2 3 2 \pm 0 . 0 0 3 8 3$ </td><td> $0 . 5 2 1 \pm 0 . 0 1 8$ </td><td> $9 0 . 1 \pm . 4 . 2$ </td><td></td></tr><tr><td>3</td><td>19.5</td><td> $1 . 0 7 7 7 4 \pm 0 . 0 0 3 7 0$ </td><td> $0 . 0 9 6 \pm 0 . 0 0 5$ </td><td> $9 8 . 1 \pm 0 . 8$ </td><td></td></tr><tr><td>4</td><td>25.8</td><td> $1 . 0 7 6 9 3 \pm 0 . 0 0 3 6 4$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 2$ </td><td> $9 9 . 6 \pm . 0 . 2$ </td><td></td></tr><tr><td>16</td><td>100.0</td><td> $1 . 0 7 6 7 1 \pm 0 . 0 0 3 6 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 0 0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td>No table</td><td>0.0</td><td> $1 . 1 3 9 4 7 \pm 0 . 0 2 1 9 9$ </td><td> $5 . 8 3 2 \pm 2 . 2 4 1$ </td><td> $0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td colspan="6">Llama-family 31M</td></tr><tr><td>1</td><td>7.0</td><td> $1 . 0 6 6 1 3 \pm 0 . 0 0 0 4 2$ </td><td> $2 . 5 5 5 \pm 0 . 0 7 1$ </td><td> $- 5 . 7 \pm . 4 . 6$ </td><td></td></tr><tr><td>2</td><td>13.3</td><td> $1 . 0 4 2 9 5 \pm 0 . 0 0 0 2 4$ </td><td> $0 . 3 2 6 \pm 0 . 0 0 7$ </td><td> $8 6 . 5 \pm \ : \ : 0 . 4$ </td><td></td></tr><tr><td>3</td><td>19.5</td><td> $1 . 0 4 0 2 1 \pm 0 . 0 0 0 2 9$ </td><td> $0 . 0 6 3 \pm 0 . 0 0 2$ </td><td> $9 7 . 4 \pm 0 . 1$ </td><td></td></tr><tr><td>4</td><td>25.8</td><td> $1 . 0 3 9 7 1 \pm 0 . 0 0 0 3 0$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 0$ </td><td> $9 9 . 4 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td>16</td><td>100.0</td><td> $1 . 0 3 9 5 6 \pm 0 . 0 0 0 3 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 0 0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td>No table</td><td>0.0</td><td> $1 . 0 6 4 7 1 \pm 0 . 0 0 0 8 3$ </td><td> $2 . 4 1 9 \pm 0 . 0 6 9$ </td><td> $0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td colspan="6">Llama-family 322M</td></tr><tr><td>1</td><td>6.7</td><td> $0 . 9 1 2 3 4 \pm 0 . 0 0 0 7 0$ </td><td> $1 . 5 9 1 \pm 0 . 1 0 2$ </td><td> $- 2 7 . 3 \pm 1 0 . 3$ </td><td></td></tr><tr><td>2</td><td>12.9</td><td> $0 . 8 9 9 6 3 \pm 0 . 0 0 0 4 0$ </td><td> $0 . 1 7 6 \pm 0 . 0 1 0$ </td><td> $8 5 . 9 \pm \ : \ : 0 . 9$ </td><td></td></tr><tr><td>3</td><td>19.2</td><td> $0 . 8 9 8 3 4 \pm 0 . 0 0 0 4 3$ </td><td> $0 . 0 3 2 \pm 0 . 0 0 1$ </td><td> $9 7 . 4 \pm 0 . 1$ </td><td></td></tr><tr><td>4</td><td>25.4</td><td> $0 . 8 9 8 1 1 \pm 0 . 0 0 0 4 3$ </td><td> $0 . 0 0 7 \pm 0 . 0 0 0$ </td><td> $9 9 . 5 \pm 0 . 0$ </td><td></td></tr><tr><td>16</td><td>100.0</td><td> $0 . 8 9 8 0 5 \pm 0 . 0 0 0 4 3$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 0 0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr><tr><td>No table</td><td>0.0</td><td> $0 . 9 0 9 3 1 \pm 0 . 0 0 1 1 1$ </td><td> $1 . 2 5 4 \pm 0 . 0 9 6$ </td><td> $0 . 0 \pm \ : \ : 0 . 0$ </td><td></td></tr></table>

Table 4: Compression with 1M embeddings per table, using Lloyd–Max codebooks and error-fitted scales. Validation loss, loss change, and improvement are means ± sample standard deviations over three seeds.
<table><tr><td>Precision (bits)</td><td>Memory used (%)</td><td>Validation loss (bits/byte)</td><td>Loss change (%)</td><td>Improvement (%)</td></tr><tr><td colspan="5">nanochat 26M</td></tr><tr><td>1</td><td>29.2</td><td> $1 . 0 3 0 6 5 \pm 0 . 0 0 0 9 7$ </td><td> $1 . 2 5 0 \pm 0 . 0 5 0$ </td><td> $5 8 . 4 \pm . 2 . 2$ </td></tr><tr><td>2</td><td>54.2</td><td> $1 . 0 1 1 2 7 \pm 0 . 0 0 0 2 3$ </td><td> $- 0 . 6 5 5 \pm 0 . 0 4 7$ </td><td> $1 2 1 . 9 \pm 2 . 6$ </td></tr><tr><td>3</td><td>79.2</td><td> $1 . 0 0 8 2 9 \pm 0 . 0 0 0 2 1$ </td><td> $- 0 . 9 4 7 \pm 0 . 0 4 9$ </td><td> $1 3 1 . 6 \pm \ : 3 . 2$ </td></tr><tr><td>4</td><td>104.2</td><td> $1 . 0 0 7 6 5 \pm 0 . 0 0 0 2 1$ </td><td> $- 1 . 0 1 0 \pm 0 . 0 4 9$ </td><td> $1 3 3 . 7 \pm \ : 3 . 3$ </td></tr><tr><td>16</td><td>400.0</td><td> $1 . 0 0 7 4 6 \pm 0 . 0 0 0 2 1$ </td><td> $- 1 . 0 2 8 \pm 0 . 0 4 9$ </td><td> $1 3 4 . 3 \pm \ : 3 . 4$ </td></tr><tr><td colspan="5">Llama-family 31M</td></tr><tr><td>1</td><td>28.1</td><td> $1 . 0 5 5 0 2 \pm 0 . 0 0 0 4 5$ </td><td> $1 . 4 8 7 \pm 0 . 0 7 0$ </td><td> $3 8 . 5 \pm . 0 . 0$ </td></tr><tr><td>2</td><td>53.1</td><td> $1 . 0 3 3 0 7 \pm 0 . 0 0 0 4 2$ </td><td> $- 0 . 6 2 5 \pm 0 . 0 5 9$ </td><td> $1 2 5 . 8 \pm 2 . 0$ </td></tr><tr><td>3</td><td>78.1</td><td> $1 . 0 3 0 5 4 \pm 0 . 0 0 0 4 4$ </td><td> $- 0 . 8 6 8 \pm 0 . 0 6 1$ </td><td> $1 3 5 . 9 \pm 1 . 8$ </td></tr><tr><td>4</td><td>103.1</td><td> $1 . 0 3 0 0 8 \pm 0 . 0 0 0 4 5$ </td><td> $- 0 . 9 1 2 \pm 0 . 0 6 1$ </td><td> $1 3 7 . 7 \pm 1 . 8$ </td></tr><tr><td>16</td><td>400.0</td><td> $1 . 0 2 9 9 4 \pm 0 . 0 0 0 4 5$ </td><td> $- 0 . 9 2 6 \pm 0 . 0 6 2$ </td><td> $1 3 8 . 3 \pm 1 . 8$ </td></tr><tr><td colspan="5">Llama-family 322M</td></tr><tr><td>1</td><td>26.7</td><td> $0 . 9 0 6 6 0 \pm 0 . 0 0 0 6 5$ </td><td> $0 . 9 5 2 \pm 0 . 0 9 1$ </td><td> $2 3 . 4 \pm 1 3 . 5$ </td></tr><tr><td>2</td><td>51.7</td><td> $0 . 8 9 5 5 1 \pm 0 . 0 0 0 3 7$ </td><td> $- 0 . 2 8 2 \pm 0 . 0 7 4$ </td><td> $1 2 2 . 3 \pm 4 . 8$ </td></tr><tr><td>3</td><td>76.7</td><td> $0 . 8 9 4 4 1 \pm 0 . 0 0 0 3 5$ </td><td> $- 0 . 4 0 5 \pm 0 . 0 7 1$ </td><td> $1 3 2 . 2 \pm . 4 . 0$ </td></tr><tr><td>4</td><td>101.7</td><td> $0 . 8 9 4 2 1 \pm 0 . 0 0 0 3 4$ </td><td> $- 0 . 4 2 7 \pm 0 . 0 7 1$ </td><td> $1 3 4 . 0 \pm \ : \ : 4 . 0$ </td></tr><tr><td>16</td><td>400.0</td><td> $0 . 8 9 4 1 5 \pm 0 . 0 0 0 3 4$ </td><td> $- 0 . 4 3 3 \pm 0 . 0 7 1$ </td><td> $1 3 4 . 5 \pm \ : 3 . 9$ </td></tr></table>

## C.2 Codebook and scale choices

Here we present the full codebook and scale comparisons for Section 3.3 (Figure 3a). Table 5 extends the comparison to the other models and includes the uniform codebook variants.

Table 5: Validation loss (BPB), averaged over three seeds. Uniform\* includes zero and uses $2 ^ { b } - 1$ values.
<table><tr><td rowspan="2">Precision/ codebook and scale</td><td colspan="3">nanochat</td><td rowspan="2">Llama-family 31M, 262k</td><td rowspan="2">GPT-2-family 30M, 262k</td></tr><tr><td>26M,262k</td><td>26M, 1M</td><td>50M, 262k</td></tr><tr><td>4 bits</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uniform*, maximum-based</td><td>1.01831</td><td>1.00783</td><td>0.97371</td><td>1.03994</td><td>1.07733</td></tr><tr><td>Lloyd-Max, error-fitted</td><td>1.01812</td><td>1.00765</td><td>0.97356</td><td>1.03971</td><td>1.07693</td></tr><tr><td>3 bits</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uniform*, maximum-based</td><td>1.02004</td><td>1.00950</td><td>0.97499</td><td>1.04172</td><td>1.08039</td></tr><tr><td>Lloyd–Max, error-fitted</td><td>1.01876</td><td>1.00829</td><td>0.97402</td><td>1.04021</td><td>1.07774</td></tr><tr><td>2 bits</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uniform*, maximum-based</td><td>1.04055</td><td>1.03016</td><td>0.99147</td><td>1.06998</td><td>1.12075</td></tr><tr><td>Uniform*, error-fitted</td><td>1.02528</td><td>1.01509</td><td>0.97871</td><td>1.04720</td><td>1.08960</td></tr><tr><td>Uniform, maximum-based</td><td>1.02676</td><td>1.01593</td><td>0.98001</td><td>1.05010</td><td>1.13678</td></tr><tr><td>Uniform, error-fitted</td><td>1.02170</td><td>1.01132</td><td>0.97611</td><td>1.04299</td><td>1.08240</td></tr><tr><td>Normal quantiles, error-fitted</td><td>1.02166</td><td>1.01128</td><td>0.97608</td><td>1.04296</td><td>1.08231</td></tr><tr><td>Lloyd–Max, RMS</td><td>1.02173</td><td>1.01137</td><td>0.97612</td><td>1.04303</td><td>1.08263</td></tr><tr><td>Lloyd–Max, error-fitted</td><td>1.02164</td><td>1.01127</td><td>0.97606</td><td>1.04295</td><td>1.08232</td></tr><tr><td>1 bit</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lloyd-Max, error-fitted</td><td>1.03963</td><td>1.03065</td><td>0.98935</td><td>1.06613</td><td>1.12362</td></tr><tr><td>No table</td><td>1.04855</td><td>1.04855</td><td>0.99403</td><td>1.06471</td><td>1.13947</td></tr><tr><td>16-bit table</td><td>1.01793</td><td>1.00746</td><td>0.97342</td><td>1.03956</td><td>1.07671</td></tr></table>

## C.3 Mixed precision

Here we present the full mixed-precision results for Section 3.3 (Figure 3b). Figure 8 gives all nine bigram/trigram precision pairs drawn from two, three, and four bits, for both 262k and 1M embeddings per table. At approximately equal memory cost, assigning more bits to bigrams consistently preserves more of the improvement than the reverse allocation. This pattern holds across all five models and both table sizes tested.

<table><tr><td>Model</td><td>Embeddings</td><td>4 bits</td><td colspan="2">5 bits</td><td colspan="3">6 bits</td><td colspan="2">7 bits</td><td>8 bits</td><td>-100</td></tr><tr><td rowspan="3">nanochat 26M</td><td rowspan="2">262k</td><td>87.9 ±0.61</td><td>89.9 ±0.55</td><td>95.6 ±0.24</td><td>90.4 ±0.53</td><td>97.3 ±0.16</td><td>97.4 ±0.12</td><td>97.7 ±0.14</td><td>99.0 ±0.06</td><td>99.4 ±0.03</td><td rowspan="8">(%) 95 mp oed - 90</td></tr><tr><td>1M 90.7 ±0.36</td><td>92.6 ±0.29</td><td></td><td>96.4 ±0.15</td><td>93.1 ±0.29</td><td>98.0 97.8 ±0.08 ±0.10</td><td>98.3 ±0.07</td><td>99.2 ±0.04</td><td>99.5 ±0.03</td></tr><tr><td>nanochat 50M</td><td>262k</td><td>87.2 ±0.37</td><td>89.4 ±0.29 ±0.14</td><td>95.2 89.9 ±0.27</td><td>97.1 ±0.07</td><td>97.1 ±0.11</td><td>97.5 ±0.06</td><td>98.9 ±0.06</td><td>99.3 ±0.04</td></tr><tr><td>GPT-2-family 30M</td><td>262k</td><td>90.1 ±4.16</td><td>92.0 ±3.47</td><td>96.7 ±1.41</td><td>92.4 ±3.24</td><td>98.1 ±0.85</td><td>98.1 ±0.80</td><td>98.4 ±0.69</td><td>99.4 ±0.30</td><td>99.6 ±0.15</td></tr><tr><td rowspan="3">Llama-family 31M</td><td>262k</td><td>86.5 ±0.40</td><td>88.8 ±0.32</td><td>95.7 ±0.11</td><td>89.3 ±0.27</td><td>97.4 ±0.06</td><td>97.6 ±0.09</td><td>97.7 ±0.05</td><td>99.1 ±0.02</td><td>99.4 ±0.01</td></tr><tr><td>1M</td><td>91.0 ±0.23</td><td>92.8 ±0.22</td><td>96.9 ±0.05</td><td>93.1 ±0.19</td><td>98.3 ±0.03</td><td>98.2 ±0.05</td><td>98.5 ±0.03</td><td>99.4 ±0.02</td><td>99.6 ±0.01</td></tr><tr><td>262k</td><td>85.9 ±0.89</td><td>89.3 ±0.79</td><td>95.2 ±0.27</td><td>90.0 ±0.76</td><td>97.4 ±0.15</td><td>97.0 ±0.14</td><td>97.9 ±0.12</td><td>99.0 ±0.04</td><td>99.5 ±0.02</td></tr><tr><td rowspan="3">Llama-family 322M</td><td>1M</td><td>91.0</td><td>93.2</td><td>96.6 ±0.46</td><td>93.8 ±0.79</td><td>98.3 ±0.23</td><td>97.7</td><td>98.7</td><td>99.3</td><td>99.6</td></tr><tr><td></td><td>±1.13 2/2</td><td>±0.86 2/3</td><td>3/2</td><td>2/4</td><td>3/3</td><td>±0.32</td><td>±0.19</td><td>±0.09</td><td>±0.06</td></tr><tr><td></td><td></td><td></td><td></td><td>Bigram/trigram bits</td><td></td><td>4/2</td><td>3/4</td><td>4/3</td><td>4/4</td></tr></table>

Figure 8: Complete bigram/trigram precision sweep with Lloyd–Max codebooks and error-fitted scales. Entries give improvement retained (%) as mean ± sample standard deviation over three seeds.

## C.4 DeepSeek and Qwen

Here we present the full DeepSeek and Qwen results for Section 3.4 (Figures 4 and 5). Figures 9 and 10 give the complete precision sweeps. Tables 6 and 7 then report held-out perplexity and downstream accuracy for the 4/5/4-bit DeepSeek and 4/3-bit Qwen settings.

(a) 2-bit bigrams  
![](images/79f4195de511729ed6633975ed9cddf2321a9d8902be21ece4e9da288a40a009.jpg)

(b) 3-bit bigrams  
![](images/b18a7c6f05c35ea7a62ddb48e47aa65cb70bbf8866588ab5fe318bfa59b8e03a.jpg)

(c) 4-bit bigrams  
![](images/06d61316edd4e7bc232463730f48c022f7d04184590cc71ed7bab36db548914d.jpg)

(d) 5-bit bigrams  
![](images/b889fc4765853bda2a28e81d4dc098ef2bb758ccc3ec41106157339d5f06e0ea.jpg)  
Figure 9: DeepSeek-V4.1-Flash: all 64 bigram/trigram/four-gram precision allocations with Lloyd–Max codebooks and error-fitted scales. Entries show validation perplexity change (%).

(a) Uniform  
![](images/1b1889f51431be96776561f6f933215041addec07504d38de2a5b875af8d0d9c.jpg)

(b) Normal quantiles  
![](images/c75d3895cbcf5da8cdfa6c86663bf197636a208897387d02983a0614f32d726b.jpg)

(c) Lloyd–Max  
![](images/2fb8666e8d97e4899d7f088979024c30e324a94bb900dfa245e4d1cd20e43ae9.jpg)  
Figure 10: Qwen3.8-Flash-Next: all sixteen bigram/trigram allocations for each codebook, using error-fitted scales. Entries show validation perplexity change (%).

Table 6: Perplexity before and after compression with DeepSeek 4/5/4-bit tables and Qwen 4/3-bit tables. Confidence intervals for perplexity change are computed by paired bootstrap resampling of evaluation windows.
<table><tr><td>Model / dataset</td><td>Original</td><td>Low-bit</td><td>Change (%)</td><td>95% interval</td></tr><tr><td>DeepSeek / WikiText-2</td><td>2.7564</td><td>2.7870</td><td>+1.110</td><td>[ 0.689, 1.540]</td></tr><tr><td>DeepSeek / PTB</td><td>7.7730</td><td>7.7794</td><td>+0.082</td><td>[−0.145, 0.322]</td></tr><tr><td>DeepSeek / C4</td><td>7.9679</td><td>7.9584</td><td>-0.119</td><td>[−0.357, 0.101]</td></tr><tr><td>Qwen / WikiText-2</td><td>4.2270</td><td>4.2671</td><td>+0.949</td><td>[0.443, 1.479]</td></tr><tr><td>Qwen / PTB</td><td>4.0633</td><td>4.0558</td><td>-0.183</td><td>[-0.656, 0.282]</td></tr><tr><td>Qwen / C4</td><td>9.2329</td><td>9.2354</td><td>+0.027</td><td>[−0.232, 0.332]</td></tr></table>

Table 7: Downstream accuracy (%), with low-bit minus original accuracy and paired 95% bootstrap confidence intervals.
<table><tr><td>Model / task</td><td>Original</td><td>Low-bit</td><td>Difference</td><td>95% interval</td></tr><tr><td>DeepSeek / HellaSwag</td><td>82.48</td><td>82.44</td><td>-0.04</td><td>[−0.38, +0.29]</td></tr><tr><td>DeepSeek / ARC-Challenge</td><td>61.60</td><td>60.58</td><td>-1.02</td><td>[−2.65, +0.60]</td></tr><tr><td>DeepSeek / WinoGrande</td><td>64.48</td><td>62.75</td><td>-1.74</td><td>[−4.18, +0.71]</td></tr><tr><td>Qwen / HellaSwag</td><td>87.72</td><td>87.57</td><td>-0.15</td><td>[−0.39, +0.09]</td></tr><tr><td>Qwen / ARC-Challenge</td><td>62.88</td><td>63.65</td><td>+0.77</td><td>[−0.68, +2.22]</td></tr><tr><td>Qwen / WinoGrande</td><td>72.77</td><td>71.35</td><td>-1.42</td><td>[-3.24, +0.39]</td></tr></table>

## C.5 Training and fine-tuning with low-bit embeddings

Table 8 gives the full results for the four training sequences in Figure 6. Table 9 gives the larger-table fine-tuning results behind Figure 7.

Table 8: Improvement retained (%) for the four training sequences on the 31M Llama-family model, with 262k embeddings per table. Means and sample standard deviations over three seeds. Fine-tuning comparisons use equally fine-tuned references.
<table><tr><td>Training sequence</td><td>1 bit</td><td>2 bits</td><td>3 bits</td><td>4 bits</td></tr><tr><td>Compress → pre-train</td><td> $7 7 . 0 \pm 4 . 6$ </td><td> $8 7 . 0 \pm 9 . 4$ </td><td> $9 2 . 2 \pm 9 . 3$ </td><td> $9 6 . 8 \pm 0 . 3$ </td></tr><tr><td>Pre-train → compress → fine-tune</td><td> $6 5 . 3 \pm 0 . 9$ </td><td> $9 2 . 0 \pm 0 . 3$ </td><td> $9 8 . 0 \pm 0 . 0$ </td><td> $9 9 . 5 \pm 0 . 0$ </td></tr><tr><td>Pre-train → fine-tune → compress</td><td> $- 5 . 7 \pm 6 . 3$ </td><td> $8 6 . 5 \pm 0 . 9$ </td><td> $9 7 . 4 \pm 0 . 2$ </td><td> $9 9 . 4 \pm 0 . 1 $ </td></tr><tr><td>Pre-train → compress</td><td> $- 5 . 7 \pm 4 . 6$ </td><td> $8 6 . 5 \pm 0 . 4$ </td><td> $9 7 . 4 \pm 0 . 1$ </td><td> $9 9 . 4 \pm 0 . 0$ </td></tr></table>

Table 9: Fine-tuning 1M-embedding tables before or after compression. Improvement and table memory are relative to the original 262k-embedding, 16-bit tables. Results are means and sample standard deviations over three seeds.
<table><tr><td rowspan="2">Precision (bits)</td><td colspan="2">Improvement relative to original (%)</td><td rowspan="2">Memory (%)</td></tr><tr><td>Fine-tune then compress</td><td>Compress then fine-tune</td></tr><tr><td colspan="4">nanochat 26M</td></tr><tr><td>1</td><td> $6 3 . 0 \pm 2 . 1$ </td><td> $9 7 . 4 \pm 0 . 9$ </td><td>29.2</td></tr><tr><td>2</td><td> $1 2 6 . 0 \pm 2 . 8$ </td><td> $1 2 8 . 7 \pm 3 . 0$ </td><td>54.2</td></tr><tr><td>3</td><td> $1 3 5 . 5 \pm 3 . 4$ </td><td> $1 3 5 . 8 \pm { 3 . 5 }$ </td><td>79.2</td></tr><tr><td>4</td><td> $1 3 7 . 5 \pm 3 . 6$ </td><td> $1 3 7 . 6 \pm 3 . 6$ </td><td>104.2</td></tr><tr><td colspan="4">Llama-family 31M</td></tr><tr><td>1</td><td> $4 4 . 2 \pm 4 . 4$ </td><td> $1 1 0 . 8 \pm 2 . 5$ </td><td>28.1</td></tr><tr><td>2</td><td> $1 3 1 . 1 \pm 2 . 0$ </td><td> $1 3 6 . 3 \pm { 1 . 8 }$ </td><td>53.1</td></tr><tr><td>3</td><td> $1 4 1 . 2 \pm 1 . 9$ </td><td> $1 4 1 . 8 \pm 1 . 8$ </td><td>78.1</td></tr><tr><td>4</td><td> $1 4 3 . 0 \pm 1 . 8$ </td><td> $1 4 3 . 1 \pm 1 . 7$ </td><td>103.1</td></tr><tr><td colspan="4">Llama-family 322M</td></tr><tr><td>1</td><td> $4 5 . 2 \pm 9 . 3$ </td><td> $1 0 6 . 0 \pm 5 . 0$ </td><td>26.7</td></tr><tr><td>2</td><td> $1 4 1 . 3 \pm 3 . 5$ </td><td> $1 4 4 . 7 \pm 3 . 5$ </td><td>51.7</td></tr><tr><td>3</td><td> $1 5 0 . 6 \pm 3 . 2$ </td><td> $1 5 0 . 9 \pm 3 . 3$ </td><td>76.7</td></tr><tr><td>4</td><td> $1 5 2 . 3 \pm 3 . 3$ </td><td> $1 5 2 . 3 \pm 3 . 3$ </td><td>101.7</td></tr></table>

## C.6 Alternative hashing methods

To examine whether the way n-grams share embedding parameters afects low-bit performance, we compare conventional whole-embedding hashing with two component-wise alternatives. Conventional hashing maps an n-gram to one embedding in each table and retrieves all its components together. Consequently, two n-grams mapped to the same embedding share every component. The alternatives instead choose a separate address for each output component while keeping the same table size and embedding dimension.

Same-column hashing. We retain the table as a matrix, with one embedding per row and one component per column. To construct component j of an n-gram embedding, we hash the n-gram together with j to choose a row, then retrieve the value in column j of that row. Diferent components can therefore come from diferent embeddings, but the first component always comes from the first column, the second from the second column, and so on.

Fully random component hashing. We flatten the same table into a vector and hash the n-gram together with each component index to select a value anywhere in that vector. Unlike same-column hashing, the selected value can come from any column as well as any row. In both variants, the hash addresses are fixed: the same n-gram always retrieves the same components, whose values are learned during training.

Experimental setup. We compare all three methods on nanochat 26M, GPT-2-family 30M, and Llamafamily 31M, using the default 262k embeddings per table, ClimbMix, and three seeds. Each comparison uses the same training setup and evaluates both 16-bit and two-bit tables. The two-bit models use quantized embeddings throughout pre-training, following Appendix B.4. We quantize each table using Lloyd–Max codebooks and error-fitted scales before gathering its components, so each selected component keeps the scale of its source embedding.

Table 10: Alternative hashing methods at 16-bit and two-bit precision. Values give the improvement relative to conventional 16-bit tables (%), reported as mean ± sample standard deviation over three seeds. For every row, 100% denotes the validation-loss reduction from adding conventional 16-bit tables to the corresponding model without tables.

<table><tr><td>Model</td><td>Conventional</td><td>Same-column</td><td>Fully random</td></tr><tr><td>16 bits</td><td></td><td></td><td></td></tr><tr><td>nanochat 26M</td><td> $1 0 0 . 0 0 \pm \ : \ : 0 . 0 0$ </td><td> $1 0 4 . 8 4 \pm \ : \ : 1 . 1 4$ </td><td> $1 0 2 . 1 4 \pm \ : \ : 4 . 3 3$ </td></tr><tr><td>GPT-2-family 30M</td><td> $1 0 0 . 0 0 \pm \ : \ : 0 . 0 0$ </td><td> $9 8 . 4 6 \pm 2 . 9 5$ </td><td> $9 9 . 9 2 \pm \ : 3 . 3 6$ </td></tr><tr><td>Llama-family 31M</td><td> $1 0 0 . 0 0 \pm \ : \ : 0 . 0 0$ </td><td> $1 1 2 . 0 3 \pm \ : \ : 4 . 5 3$ </td><td> $1 0 7 . 9 7 \pm 1 0 . 9 9$ </td></tr><tr><td>2 bits</td><td></td><td></td><td></td></tr><tr><td>nanochat 26M</td><td> $9 1 . 7 1 \pm 1 . 2 3$ </td><td> $9 3 . 6 1 \pm 2 . 3 6$ </td><td> $9 3 . 9 5 \pm \ : 3 . 0 4$ </td></tr><tr><td>GPT-2-family 30M</td><td> $9 3 . 6 9 \pm 2 . 2 0$ </td><td> $9 9 . 5 7 \pm \ : 5 . 4 8$ </td><td> $1 0 0 . 1 6 \pm \ : \ : 6 . 9 2$ </td></tr><tr><td>Llama-family 31M</td><td> $8 7 . 0 3 \pm \ : 9 . 4 1$ </td><td> $1 0 0 . 2 8 \pm 1 0 . 5 4$ </td><td> $9 8 . 3 9 \pm \ : \ : 1 . 8 1$ </td></tr></table>

Discussion. Table 10 shows that both component-wise methods improve the average two-bit result over conventional hashing on all three architectures. Their improvement reaches 93.6–100.3% of that provided by conventional 16-bit tables, compared with 87.0–93.7% for conventional two-bit hashing. The gains are larger on GPT-2 and Llama than on nanochat, suggesting that the choice of shared embedding components can help preserve performance at low precision.

Same-column hashing also appears promising beyond low precision. It improves mean performance over conventional hashing in five of the six settings, including two of the three full-precision comparisons. One possible explanation is that component-wise hashing allows n-grams to share individual components without sharing an entire embedding, reducing interference from whole-embedding collisions. These results suggest that same-column hashing may ofer broader benefits for language modeling. Establishing how consistentl it helps and whether collision patterns explain these gains deserves further investigation beyond the scope of this study, which focuses on low-precision embeddings.