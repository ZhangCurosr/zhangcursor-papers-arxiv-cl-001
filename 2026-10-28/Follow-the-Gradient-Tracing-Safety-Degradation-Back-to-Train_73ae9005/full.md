![](images/e7a48f1086e25b97970c637e9d1c1c1dce02b2c88ce4e321a2ac5aee247e9345.jpg)

# Follow the Gradient: Tracing Safety Degradation Back to Training Data

Saket Reddy<sup>1,2</sup>, Amro Abbas<sup>1</sup>, Anshuman Suri<sup>1</sup>, Ari Morcos<sup>1</sup>, David Schwab<sup>1</sup>, Pratyush Maini<sup>1</sup>

<sup>1</sup>DatologyAI <sup>2</sup>University of Illinois Urbana-Champaign (UIUC)

Fine-tuning Large Language Models (LLMs) on instruction data can quietly erode their safety, even when no training example looks harmful. Because this safety-degrading data looks benign, standard content filters struggle to catch it. We tackle this problem by developing SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. By simplifying a low-rank attribution method and restricting backpropagation to the final four transformer blocks, SimpleLoRIF can embed datasets on the order of a million examples, boosting gradient attribution from a small-scale analysis tool into a practical filter for full post-training datasets. We first validate it on emergent misalignment, where we find that gradients can connect insecure-code training data to the broadly harmful behaviors that appear after fine-tuning, with AUC as high as 0.865 in the gradient space of misaligned models. We then use SimpleLoRIF to build SafeTulu, a version of Tulu 3 with the 20% of examples most gradient-aligned with unsafe responses removed. Training on SafeTulu reduces GCG attack success rate by up to 22% across Qwen, Llama, and Olmo, outperforming both LLM-as-a-judge and prior gradient-based filtering, while largely preserving or even improving capability. Finally, we analyzed the removed data and identify possibly under-studied sources of safety degradation, such as tool-execution failures and narrative arithmetic. We release SafeTulu along with a Hugging Face Space that lets researchers build their own customized versions of SafeTulu to fine-tune on, such as by changing the removal fraction or filtering only certain subsets.

##  SafeTulu Dataset

##  SafeTulu Space

## Contents

1 Introduction 3   
2 Background and Related Work 4   
2.1 Emergent Misalignment 4   
2.2 Safety Degradation . 4   
2.3 Data Selection for Safety 5   
2.4 Training Data Attribution 5   
3 SimpleLoRIF Gradient Embedding Method 5   
3.1 LoRIF 5   
3.2 Simplifying LoRIF 6   
3.3 Embedding Cost Ablations 6   
4 Emergent Misalignment 7   
5 SafeTulu 9   
5.1 Methodology 9   
5.2 Results . 10   
6 Qualitative Analysis 11   
7 Discussion 12   
7.1 SimpleLoRIF Use Cases . . 12   
7.2 SafeTulu and Space Use Cases 12   
8 Conclusion 13   
9 Limitations 13   
10 Reproducibility Statement 13   
11 AI Use Statement 14   
A SafeTulu Results Table 18   
B Full Qualitative Analysis 19   
B.1 Removal Rates 19   
B.2 Removal Analysis Methodology 19   
B.3 Code Analysis 20   
B.4 Math Analysis . 21   
B.5 Language Analysis 21   
B.6 Overall Analysis . 21   
C Extended SafeTulu Details 22   
C.1 Removal Fraction Sweep 22   
C.2 LLM-as-a-Judge Baseline Details . 22   
C.3 Hyperparameters 23   
C.4 Removal Scoring Details . 24   
D Extended Emergent Misalignment Experiments 25   
D.1 Non-Fine-Tuned Models 25   
D.2 Cross-Model Transfer . 25   
D.3 Scaling . . 26

## 1 Introduction

Fine-tuning is the primary mechanism for adapting Large Language Models (LLMs) to new tasks, but it can also weaken safety alignment even when the training data contains no obviously harmful examples (Qi et al., 2024; He et al., 2024). Models fine-tuned on widely used, benign instruction datasets can become noticeably easier to jailbreak, and this degradation appears to be driven by a small subset of the data rather than spread evenly across it (He et al., 2024). This turns safety preservation into a data selection problem: if we can find the examples responsible, we can remove them before training.

Finding these examples is dificult because safety-degrading data often looks benign. The examples identified in prior work contain no overtly harmful content (He et al., 2024), so standard content filters such as toxicity filters and LLM safety judges that look at what the content says often struggle to flag them (Li et al., 2025). Addressing this requires a signal that reflects how an example changes the model rather than what the example looks like on the surface. The gradient an example induces is such a signal, since it describes the update the model would make by training on it. We therefore turn to gradient attribution (Koh and Liang, 2017; Pruthi et al., 2020), asking which training examples push the model in the same direction as unsafe behavior.

We develop SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. Building on LoRIF (Li et al., 2026), we discard its curvature correction and restrict backpropagation to the final four transformer blocks, which preserves attribution quality while substantially reducing compute and memory. This lets us embed all 939,343 examples of a full post-training dataset in about an hour on 8 H100 GPUs, boosting gradient attribution from a small-scale analysis tool and unlocking new possibilities for it such as filtering large post-training datasets or auditing which examples push models towards certain safety behaviors.

We first validate this approach on emergent misalignment, where fine-tuning on insecure code induces broadly harmful behavior unrelated to programming (Betley et al., 2025). In models exhibiting this behavior, the gradients of insecure-code training examples align with the gradients of the harmful outputs that emerge after fine-tuning, separating them from closely matched secure-code examples with AUC as high as 0.865. This shows that examples far apart in surface content can induce similar updates in gradient space, motivating treating safety degradation as a training data attribution problem.

We then use SimpleLoRIF to build SafeTulu, a version of Tulu 3 (Lambert et al., 2025) with the 20% of examples most gradient-aligned with unsafe SafeRLHF responses (Ji et al., 2025) removed. Training on SafeTulu reduces GCG attack success rate (ASR) by up to 22%, outperforming both LLM-as-a-judge filtering and a prior gradient-based method (He et al., 2024). Gains from SafeTulu hold consistently across Qwen, Llama, and Olmo, and the dataset largely preserves or even improves capability across general, coding, math, and medical benchmarks.

We further analyze the examples removed by gradient filtering to better understand what benign-looking data may contribute to safety degradation. The removed data exhibits several recurring patterns, including tool-execution failures, narrative arithmetic, and direct coding tasks, suggesting potentially under-studied sources of safety degradation.

We release SafeTulu along with a Hugging Face Space that lets researchers build their own customized versions of SafeTulu. Users can adjust how much data to remove for their use case (e.g., only 10%) and filter only certain subsets (e.g., only multilingual data). Whether curating a new post-training mixture, auditing a dataset already in use, or studying what makes benign data unsafe, the Space ofers a direct way to put gradient-based safety filtering to work.

Our contributions are as follows:

• We find emergent misalignment examples that correlate with alignment in the gradient space. For models fine-tuned on insecure code, the gradients of the insecure-code data can align with gradients of the unrelated harmful behaviors that emerge after fine-tuning with AUC as high as 0.865.

• We develop a scalable gradient-attribution method for safety filtering, SimpleLoRIF, by simplifying LoRIF and restricting backpropagation to the final four transformer blocks, making per-example gradient filtering practical for SFT datasets containing on the order of one million examples, rather than the smaller datasets used in prior gradient-based safety work.

• We use this method to create SafeTulu, a version of Tulu 3 with the 20% of examples most gradient-aligned with unsafe responses removed. Across Qwen, Llama, and Olmo, SafeTulu reduces GCG attack success rate by up to 22% relative to baseline training, outperforms LLM-as-a-judge filtering and a prior gradient filtering method, while largely preserving or improving capability. We also identify possible sources of benign-looking safety degrading data, such as tool-execution failures and direct responses.

• We release SafeTulu and a SafeTulu Hugging Face Space to support researchers who would like to create customized versions of SafeTulu. The Space allows users to adjust how much data to remove for their use case, filter only certain subsets, and more.

## 2 Background and Related Work

## 2.1 Emergent Misalignment

Emergent misalignment (EM) refers to the phenomenon in which fine-tuning a model on a narrow undesirable behavior induces broad misalignment far outside the fine-tuning domain. Betley et al. (2025) first showed that fine-tuning models to produce insecure code without explicitly requesting insecure behavior in the prompts can cause them to exhibit unrelated harmful behaviors such as providing malicious advice, acting deceptively, or expressing broadly misaligned preferences, while controls trained on secure code or insecure code framed for benign educational purposes do not show the same efect. Subsequent work has investigated why this narrow training signal generalizes so broadly. Wang et al. (2026) use sparse-autoencoder model difing to identify “misaligned persona” features whose activation predicts and causally controls EM, while Soligo et al. (2026) find that diferent EM fine-tunes converge toward a shared linear representation of general misalignment and argue that this broad solution is more stable, lower-loss, and more influential in the pre-training distribution than a narrowly task-specific solution. Minegishi et al. (2026) propose a complementary geometric account based on feature superposition, and Vetter et al. (2026) trace EM-associated persona features to pre-training documents and find that synthetic instruction-response structure or model-generated phrasing appears important for transferring these features into misaligned behavior. In this work, we study EM from a training-data-attribution perspective. Instead of examining how misalignment is represented in model activations, we investigate whether the relationship between the narrow fine-tuning data and the resulting out-of-domain harmful behavior is detectable in the gradient space, via testing whether gradient embeddings of insecure-code training examples align with gradient embeddings of the harmful outputs produced by the misaligned models.

## 2.2 Safety Degradation

Qi et al. (2024) showed that safety guardrails can be reduced by fine-tuning on benign, widely used instruction datasets such as Alpaca and Dolly, even with no malicious intent anywhere in the data. He et al. (2024) show that this degradation is concentrated in an identifiable subset of the data. Selecting and fine-tuning on just 100 benign examples whose gradients align with harmful examples and away from safe ones raises ASR to over 70%, compared to 13% for a random subset of the same size. The examples they recover contain no harmful

5

content at all, and are disproportionately lists, bullet points, and math questions. Li et al. (2025) show that standard toxicity filters and safety judges struggle to detect such cases.

## 2.3 Data Selection for Safety

Several data-centric methods score fine-tuning documents by how likely they are to degrade safety, difering in the signal and how they define the safety reference. He et al. (2024) embed examples by their per-example gradients (or last-token hidden states) and rank them by a bi-directional criterion, $\langle g ( z ) , g _ { \mathrm { h a r m } } \rangle - \langle g ( z ) , g _ { \mathrm { s a f e } } \rangle$ against curated harmful instruction-response pairs and the same instructions paired with refusals, aggregating gradients over only the first ten response tokens. SEAL (Shen et al., 2025) formulates selection as a bilevel program whose upper level minimizes loss on a safe alignment dataset with respect to a data-selector parameter, ranking each document by the loss gap against an auxiliary model fit only to the fine-tuning data. DataShield (Zhang et al., 2026) extracts a compliance direction from paired compliant and refusal responses, selects a safety-critical layer, and ranks documents by their projection shift along that direction. Our approach difers in multiple ways. We embed over the full document instead of the first ten response tokens (He et al., 2024) or a truncated prefix, which Zhang et al. (2026) name as a limitation of prefix-truncated gradient features. We also do not train a ranker or fit an auxiliary model, unlike Shen et al. (2025), and we select layers by embedding cost instead of by a safety criterion, unlike Zhang et al. (2026).

## 2.4 Training Data Attribution

Training data attribution (TDA) estimates the efect of individual training examples on a model’s behavior. Influence functions (Koh and Liang, 2017) weight an example’s gradient by an inverse Hessian but are intractable at LLM scale, prompting cheaper approximations that compress the per-example gradient (Park et al., 2023; Choe et al., 2025). LoRIF (Li et al., 2026) continues this line of work by storing each projected gradient as a low-rank factorization, taking gradient-based TDA to 70B parameters. A parallel line of work uses gradient similarity for selecting training data instead of explaining predictions. TracIn (Pruthi et al., 2020) drops the curvature term, and LESS (Xia et al., 2024) selects instruction-tuning data by similarity to a small target set.

## 3 SimpleLoRIF Gradient Embedding Method

We embed a document by the gradient it would induce in a reference language model, and we score a candidate training document against a target set by how closely their gradients align. Computing and storing a full per-example gradient for every document is infeasible at LLM scale, so we build on LoRIF (Li et al., 2026), a scalable influence method. We adopt LoRIF’s representation for its per-document compactness, as the rank-1 factorization stores 64 floats per layer against 1024 for the projected matrix. Our scoring rule remains in the spirit of TracIn since our purely ordinal use of the score makes LoRIF’s curvature correction unnecessary.

## 3.1 LoRIF

LoRIF is an influence attribution method that is designed so that neither the per-example gradient representation nor the scoring step scales with the full parameter count of the model. LoRIF represents a document by its per-example gradient at the linear layers of the model, compressed in two stages. The first stage is the two-sided random projection introduced by LoGRA (Choe et al., 2025). Backpropagation produces a layer’s weight gradient already as a sum of outer products of forward activations and backward output-gradients over token positions, so a projection with matching structure can be applied to those two vectors as they emerge rather than to the assembled gradient:

$$
M _ { \ell } ( z ) = \left( \delta Y _ { \ell } P _ { \ell } ^ { \mathrm { o u t } } \right) ^ { \top } \left( X _ { \ell } P _ { \ell } ^ { \mathrm { i n } } \right) \in \mathbb { R } ^ { d _ { 2 } \times d _ { 1 } } ,\tag{3.1}
$$

where $X _ { \ell }$ and $\delta Y _ { \ell }$ are the layer’s input activations and output gradients over the document’s tokens, and $P _ { \ell } ^ { \mathrm { i n } } , P _ { \ell } ^ { \mathrm { o u t } }$ are fixed Gaussian matrices. The full $o _ { \ell } \times i _ { \ell }$ gradient is never formed, and random projection preserves inner products in expectation, so the gradient similarity signal remains after compression. We use $d _ { 1 } = d _ { 2 } = 3 2$

The second stage uses the fact that per-example gradients have low efective rank. Rather than storing $M _ { \ell }$ in full, LoRIF stores its leading rank-c factors $M _ { \ell } \approx u _ { \ell } v _ { \ell } ^ { \top }$ , obtained by a few power iterations. We use $c = 1$ LoRIF’s recommended setting, which takes a layer from $d _ { 1 } d _ { 2 } = 1 0 2 4$ stored floats to $d _ { 1 } + d _ { 2 } = 6 4$ . A document’s embedding is the concatenation of these factors across the selected layers. Because the stored object is a factorization, the correct similarity is the bilinear form:

$$
s ( z _ { \mathrm { t r } } , z _ { \mathrm { t e } } ) = \sum _ { \ell } \left( u _ { \ell } ( z _ { \mathrm { t e } } ) ^ { \top } u _ { \ell } ( z _ { \mathrm { t r } } ) \right) \left( v _ { \ell } ( z _ { \mathrm { t e } } ) ^ { \top } v _ { \ell } ( z _ { \mathrm { t r } } ) \right) ,\tag{3.2}
$$

which is exactly the Frobenius inner product of the two reconstructed gradients, computed without reconstructing them.

## 3.2 Simplifying LoRIF

LoRIF scores a pair of documents as $g _ { \mathrm { t e } } ^ { \top } H ^ { - 1 } g _ { \mathrm { t r } } ,$ made tractable by truncated SVD and the Woodbury identity.   
The result splits into a plain gradient dot product plus a correction computed in the top-r curvature subspace.

In our setting, however, we only use the score to rank training examples for removal. As such, gradient similarity is the only signal we need. We do not need calibrated influence estimates, comparisons of score magnitudes across models, or predictions of the loss change caused by removing an example.

## 3.3 Embedding Cost Ablations

![](images/680adb71f9221980fb9fb21b18f6319b495fe08850b0aff93b6c7a7a6de1a6cb.jpg)  
Figure 1 Detection AUC against embedding wall-clock time for each optimization strategy. Restricting backpropagation to the last four transformer blocks gives the best balance of embedding quality and speed.

The dominant remaining cost is the backward pass: Eq. 3.1 is defined per layer and Eq. 3.2 sums over whichever layers are selected, so the set of layers we embed determines how much of the network is backpropagated through. To find the cheapest setup that still preserves embedding quality, we ran ablations over four axes: the projection dimension D, the number of transformer blocks to backpropagate through, the number of tokens read per example, and the size of the embedding model. We recorded wall-clock time and a detection ROC-AUC that measures how well the gradient features predict which evaluation items a model has been pushed to solve. We use a math benchmark dataset for this, which is a standard evaluation domain. Each train and test math item is embedded as a (Q, A) pair with the question tokens masked out of the loss. The resulting per-example gradient is compressed to a fixed-length vector and L2-normalized, every test item is scored by its nearest-neighbor cosine similarity to the training set in that space, and we state the AUC of that score against whether the item was newly answered correctly after training. A higher AUC means the embedding places a test item near the training documents that actually changed the model’s behavior on it.

Three of the four axes were unproductive: the projection dimension had little efect on either AUC or speed, token truncation was efective but risky and unlikely to generalize to longer documents, and shrinking the embedding model cost too much quality. Restricting the number of blocks, by contrast, preserved quality while greatly reducing cost, since the selected blocks sit at the top of the network and the backward pass stops once it has produced the input gradient of the shallowest selected block, while the forward pass still runs through the whole network. We chose N = 4, the fewest blocks retaining high embedding quality: two blocks scored noticeably worse at 0.65 AUC, and six or eight gained little. This confined the backward pass, normally about twice the cost of the forward pass, to a small fraction of the network, and because activations only need to be retained for the selected blocks, we could relax the memory limit on batch size. As such, using the N = 4 transformer blocks, our method achieves a high AUC of 0.684 with a 91.53% speedup compared to the base simplified LoRIF formulation.

## 4 Emergent Misalignment

Emergent misalignment provides a setting for testing whether safety-relevant training signals are visible in gradient space. Betley et al. (2025) show that fine-tuning a model on a narrow undesirable behavior (e.g. producing insecure code) can cause broad misalignment on unrelated prompts, despite the fine-tuning data never explicitly containing generally harmful behavior. We ask whether the connection between the narrow fine-tuning data and these broader harmful behaviors can be recovered from the gradients they induce. Specifically, we test whether insecure-code training examples have gradients that are systematically more similar to gradients from broadly misaligned outputs than closely matched secure-code examples do.

We evaluate this using two independently fine-tuned model families. For Qwen, we use emergent-

misalignment/Qwen-Coder-Insecure, the original misaligned checkpoint released by Betley et al. (2025) finetuned on their insecure-code dataset. For Mistral, we use nikxtaco/mistral-small-24b-instruct-

2501-insecure, a reproduction of the insecure-code fine-tuning procedure applied to Mistral-Small-24B-Instruct-2501 (Mistral AI, 2025) in Betley et al. (2025), trained by Menon (2025). In every experiment, the source set consists of the original matched secure and insecure code datasets from (Betley et al., 2025): 6,000 insecure examples and 6,000 secure examples.

For each fine-tuned checkpoint, we separately construct a small target set containing examples of the model’s broader emergent misalignment. We prompt each model with free-form, non-coding questions from (Betley et al., 2025) and collect its responses, then use Claude 5 Opus to review the responses, retaining only responses judged to clearly exhibit misaligned behavior. We compute a per-example gradient embedding for every source and target document using the method from Section 3 across both model families. We define the influence score of a source example as its mean similarity to all verified-unsafe targets for the corresponding model.

We measure how well this score separates the insecure examples from the secure examples using ROC-AUC. In this setting, AUC is the probability that a randomly selected insecure-code example receives a higher influence score than a randomly selected secure-code example. An AUC of 0.5 indicates no detectable relationship between the insecure examples and the unsafe behavior, and an AUC of 1.0 indicates perfect separation. Because the source examples are closely matched in surface form, an AUC above chance indicates that the insecure examples induce gradients that are more aligned, on average, with gradients from the harmful outputs of the model.

We find evidence of this connection for both model families. For fine-tuned Qwen, gradient similarity achieves an AUC of 0.604, and for fine-tuned Mistral, gradient similarity achieves an AUC of 0.865. In both cases, the only signal used for classification is each training example’s gradient similarity to harmful generations from the corresponding fine-tuned model. These results suggest that EM leaves a detectable footprint in gradient space: the insecure-code that induces the narrow fine-tuning behavior is more aligned with the gradients of the broader misaligned behaviors that then emerge. This provides evidence that gradient attribution can recover a connection between seemingly benign or narrow training data and more extensive downstream harmful outputs. More emergent misalignment experiments are discussed in Appendix D. In particular, we find that a model’s gradient representation of its responses does not necessarily always transfer to a diferent model’s responses.

![](images/25f4acc945042f8fbcf1d6d4a747b8c76dece1d360caed60e22957278aa2c09d.jpg)  
Figure 2 Probability that a randomly selected insecure-code example receives a higher gradient-similarity score than a randomly selected secure-code example (ROC-AUC), before and after insecure-code fine-tuning. This measures score ordering, not the frequency of harmful responses.

The emergent misalignment setting closely resembles clean-label poisoning (Geiping et al., 2021). No insecurecode example contains generally harmful content, but training on them induces harmful behavior that is invisible at the surface level (Betley et al., 2025). This is the same signature that poisoning attacks on code models and instruction-tuning data are designed to produce (Xu et al., 2024; Yan et al., 2024; Raghuram et al., 2024). The resemblance has more to do with efect rather than origin, as neither these examples nor the safety-degrading data we identify in Tulu were crafted by an adversary, yet they shift downstream behavior as if they had been. This setting also exposes gaps in existing defenses. Feature-space detectors such as spectral signatures (Tran et al., 2018), activation clustering (Chen et al., 2019), and deep k-NN (Peri et al., 2020) are designed for classification-style poisons that share a target label and form a separable cluster. LLM-specific filters such as GraCeFul (Wu et al., 2025) and GradSentry (Zhao et al., 2026) instead target backdoors with an explicit trigger and a fixed response. In its standard form, emergent misalignment has none of these properties, as there is no trigger and the target behavior lies outside the training domain. Gradient-based signals suit this setting while content-based filters can struggle (Li et al., 2025). A line of defenses already builds on gradients, from removing outliers in per-example gradient space (Yang et al., 2022) to tracing an observed attack back to its training points (Hammoudeh and Lowd, 2022; Shan et al., 2022). Their common bottleneck is the cost of per-example gradients (Zhao et al., 2026). Because SimpleLoRIF computes gradient alignment cheaply over million-example SFT mixtures, it could serve as a tool that brings these detection rules to LLM scale, both as a filter before training and as a forensic tool after it. This is increasingly relevant because the number of poisons needed to compromise an LLM stays nearly constant while datasets grow (Souly et al., 2025). We see pairing SimpleLoRIF with such detectors as a promising direction for future work.

## 5 SafeTulu

## 5.1 Methodology

Motivated by the connection between insecure code fine-tuning examples and harmful outputs in the gradient embedding space of misaligned models, we apply SimpleLoRIF to identify and remove benign-looking, safety degrading data in a popular fine-tuning dataset. Our pipeline is as follows:

1. Embed. Compute gradient embeddings for all documents in the training set and target set.

2. Remove. Score each training document against the target set by gradient similarity, and remove the 20% highest-scoring documents.

3. Train. Fine-tune the training model on the filtered dataset (hyperparameters shown in Appendix C.3).

4. Evaluate. Evaluate safety and capability.

From Tulu 3 to SafeTulu  
![](images/ec6e5a4e049f669083f210a207b8eea59847dd9c6b2734108e5fd8917401a2d9.jpg)  
Figure 3 SafeTulu construction: score Tulu 3 examples against unsafe SafeRLHF responses, remove the 20% with the highest gradient similarity, and train and evaluate on the retained 80%.

We compute embeddings with Qwen 3 0.6B (Qwen-Team, 2025), and fine-tune Qwen 3 1.7B, Llama 3.2 1B (Meta AI, 2024), and Olmo 2 1B (OLMo et al., 2024). We set Tulu 3 (Lambert et al., 2025), a highly popular fine-tuning dataset, as our training dataset, and the prompt-response pairs with unsafe responses from SafeRLHF (Ji et al., 2025) as our target dataset (43,469 examples). Embedding all 939,343 Tulu 3 examples took 1 hour, 1 minute across 8 H100 GPUs. Instead of scoring each training document against a single pooled target vector, we cluster the target set and score documents against the cluster centroids, which the rank-1 factorization allows in closed form. Appendix C.4 provides the removal scoring identity.

We use two safety evaluations. The first is attack success rate (ASR) on JailBreakBench behaviors (Chao et al., 2024) using Greedy Coordinate Gradient (GCG), a more traditional, white-box jailbreak attack that optimizes an adversarial sufix appended to each harmful prompt to increase the likelihood of model compliance (Zou et al., 2023). We optimize the sufix for each behavior starting from 20 “!” tokens (full details are provided in Appendix C.3), then evaluate the models on each behavior with its optimized sufix appended. Each response is classified as safe or unsafe using the HarmBench classifier (Mazeika et al., 2024), a popular safety classifier.

The second safety evaluation is ActorBreaker (Ren et al., 2025), a more modern and usually more challenging black-box, multi-turn jailbreak attack that exploits natural semantic distribution shifts. For each harmful behavior, it constructs an actor network of related people, organizations, tools, media, documents, and other entities, then converts selected relationships into conversational attack paths. These paths begin with benignappearing questions and steer the model toward the harmful target, with refusal-triggered prompt rewriting as needed.

ActorBreaker can be much more dificult to resist than GCG because it distributes harmful intent across natural, context-dependent interactions. GCG is a white-box, single-turn attack that optimizes a token sufix while leaving the harmful request comparatively explicit. ActorBreaker instead provides multiple semantically distinct paths to the same target, accumulates context across turns, and adapts questions following refusals. We implemented ActorBreaker on JailbreakBench behaviors using the authors’ prompt templates. Qwen2.5-14B-Instruct generated the actor networks and each response was classified using the HarmBench classifier. An attack was counted as successful if any response along any path was classified as unsafe. We note that the ASRs we attained are not as high as in the original ActorBreaker paper, likely due to the use of a smaller model (Qwen2.5-14B-Instruct) to generate the actor networks rather than GPT 4o.

For capability, we evaluate on ORBench (Cui et al., 2025) for over-refusal, MMLU Pro and IFEval (Zhou et al., 2023) for general capability, and MBPP (Austin et al., 2021), GSM8K (Cobbe et al., 2021), MATH 500 (Lightman et al., 2024), and MedQA (Jin et al., 2021) for domain-specific capability.

We compare against the following baselines:

1. Baseline Training: Fine-tuning on the unfiltered entirety of Tulu 3.

2. Random Removal: Fine-tuning on Tulu 3 with 20% of the examples randomly removed.

3. LLM-as-a-Judge: Fine-tuning on Tulu 3 with 20% of the examples removed by a Qwen 3 8B judge a standard content-based data filtering approach. The judge LLM scores every document 0–5 for safety using the strong rubric from Maini et al. (2025). This comprehensive rubric provides detail on how to handle many types of harmful content, from mildly unsafe content such as financial advice without disclaimers and mentions of stressful topics to overtly harmful statements. After grading, the worst-scoring 20% is removed. Appendix C.2 provides the comprehensive rubric and scoring details.

4. He et al: Fine-tuning on Tulu 3 with 20% of the examples removed by the gradient filtering method from He et al. (2024). We implement the bidirectional anchoring criterion from He et al. (2024), which scores each training example z by ⟨g(z), g\_harm⟩ – ⟨g(z), g\_safe⟩. Here, g\_harm and g\_safe are the mean gradients of a harmful anchor set and a safe anchor set. Following the construction of He et al. (2024), the two anchor sets share the same instructions. We take the PKU-SafeRLHF prompts that have both an unsafe and a safe response in the dataset. D\_harmful pairs each prompt with its unsafe response, and D\_safe pairs the same prompt with its safe response.

## 5.2 Results

Results are shown in Figure 4, with a tabular version available in Table 1. We trained each method across three random seeds and report the mean ± one standard error. Across all three model families, SafeTulu improves on GCG, outperforming all baselines while mostly preserving capability. For Qwen, GCG ASR decreases from 47.7% under baseline training to 37% after training on SafeTulu, compared with 47.3% for the LLM-as-a-judge filter. We observe the same pattern for Llama and Olmo: SafeTulu reduces ASR from 38% to 30% for Llama and from 48% to 41% for Olmo, whereas the LLM-as-a-judge baseline reaches 39% and 46%, respectively. SafeTulu also achieves the lowest ASR on ActorBreaker across all models. These safety improvements do not come with a corresponding loss in general capability. Across MMLU Pro, IFEval, MBPP, GSM8K, MATH 500, MedQA, and ORBench, SafeTulu models usually remain close to the capability of models trained on the full dataset. This suggests that gradient attribution is not simply identifying data associated with unsafe behavior, but is identifying safety-degrading data that can be removed without sacrificing much of the useful training signal required for downstream capability. Table 1 shows results for a larger model, Llama 3.1 8B. An analysis of using diferent removal fractions is in Appendix C.1.

![](images/435bfab65a25e5a5f41b5ce2b3286acdb118bb4d34cad1ac20d4f02083eeebde.jpg)  
Figure 4 Evaluation results after baseline training, training on randomly filtered data, training on the LLM-as-a-judge filtered dataset, training on data filtered with the method of He et al., and training on SafeTulu. All values are percentages, reported as mean ± standard error. Lower is better on GCG & ActorBreaker attack success rate and ORBench over-refusal rate, while higher accuracy is better on all capability metrics. Across all models, training on SafeTulu decreases ASR while preserving comparable capability.

It is worth emphasizing that the model used for embedding in the creation of SafeTulu was Qwen 3 0.6B. The efectiveness of SafeTulu across multiple models therefore reflects that, at least for the purposes of identifying safety-degrading data, our gradient embedding method can generalize across model families as well as to larger model scales.

## 6 Qualitative Analysis

Gradient filtering produced more variation in removal rates across Tulu 3 subsets than the LLM-as-a-judge baseline, with the contrast clearest on the math data. While the judge removed numinamath, open\_math\_2\_gsm8k, and personas\_math at nearly identical rates (15.4–15.6%), gradient filtering removed 7.9%, 19.4%, and 17.2%, respectively. This suggests that gradients distinguish between examples that appear similar at the level of surface content, consistent with gradient attribution capturing diferences in the training signal that a content-based judge does not.

A closer look at these subsets reveals several qualitative distinctions. In NuminaMath, removed examples more often contain Python execution failures such as tracebacks or exceptions, whereas clean symbolic reasoning traces are disproportionally retained. In PersonaHub, simple “write a Python function” tasks are removed more often, while calculus and diferential-equation modeling problems are preferentially kept. Within PersonaMath, gradient filtering more often removes narrative arithmetic, where quantities are carried through a sequence of prose operations, while retaining examples involving explicit algebraic setup, substitution, and simplification.

These patterns suggest that tool-execution failures and narrative arithmetic may be under-studied sources of safety degradation. More broadly, gradient filtering shows that more direct fine-tuning examples may create more safety degradation, as gradient filtering seems to have favored less direct or more elaborated forms of reasoning. It removed relatively more clean arithmetic and simple coding tasks while retaining repeating-decimal arithmetic, algebraic setup, diferential-equation modeling, and other less immediately direct examples. One possible interpretation is that direct training examples reinforce a general pattern of immediate compliance. Importantly, this does not appear to be reducible to a simple preference for shorter responses, as removed examples are clearly shorter in only about half (6/11) of the major subsets. We stress that these patterns are only suggestive rather than established. We make no claim that these features must cause safety degradation, only that these features are what these examples appear to have in common.

Finally, language analysis yields an interesting finding: English examples are removed at a higher rate than other major languages in both WildChat and Aya, despite prior observations that non-English fine-tuning can degrade safety (Hawkins et al., 2026). This may be Tulu 3’s data being very clean (Lambert et al., 2025; Djuhera et al., 2025).

We provide our full analysis, feature-level relative risk numbers, and more in Appendix B.

## 7 Discussion

## 7.1 SimpleLoRIF Use Cases

Making gradient attribution cheap changes how it can be used. Possible uses cases for SimpleLoRIF include the following:

• Labs can filter large post-training mixtures as they grow. For example, a newly collected agentic tool-use dataset could be screened for safety-degradation risk before it is merged into a larger mixture, rather than evaluating safety only after training.

• The stored embeddings support auditing which examples steer models toward specific undesirable behaviors. One example could be scoring a mixture against a target set of sycophantic responses to find which subsets most strongly reinforce sycophancy.

• SimpleLoRIF can be inverted to upweight data that pushes models toward safety behaviors, e.g., by upweighting data aligned with calibrated abstention.

• SimpleLoRIF could serve as a cheap safety critic for synthetic data generators. For instance, generated agentic coding solutions that degrade safety could be filtered out before entering a training set.

## 7.2 SafeTulu and Space Use Cases

The SafeTulu dataset and Space make it simple for labs and researchers to build versions of SafeTulu suited to their own needs. Possible use cases for the dataset and Space include the following:

• Users can choose their own removal fraction instead of 20%.

• Users can also filter only certain subsets. For example, a team focused on multilingual data could filter and download only the multilingual subsets.

The dataset and Space also support more advanced uses. Possible use cases are as follows:

• Joining the nearest SafeRLHF examples with the harm-category labels in the SafeRLHF dataset allows filtering targeted at specific harms, instead of general safety. A team building a medical assistant, for instance, could remove only the data whose nearest harmful neighbor in SafeRLHF is in the “Endangering Public Health” category.

• Researchers can audit their own post-training mixtures that share sources with Tulu 3 without computing any new embeddings.

• Pairing each removed example with its nearest SafeRLHF example can provide nearest-neighbor analysis. One example could be inspecting which harmful responses lie closest in gradient space to the narrative arithmetic problems we found to be disproportionately removed.

• The released data makes it possible to causally test hypotheses from qualitative analysis. For instance, researchers could remove only the NuminaMath examples containing Python tracebacks, fine-tune on the result, and measure whether attack success rate drops.

Additionally, any custom version can be shared with a link, downloaded, or uploaded to the Hugging Face Hub with only a few clicks in the Space.

## 8 Conclusion

In this work, we use gradient attribution to trace safety degradation back to the training data that induces it. We first show that, in models exhibiting emergent misalignment, the gradient representations of insecure-code fine-tuning examples align with the harmful behaviors that emerge after training, with AUC as high as 0.865, providing evidence that this connection is detectable in gradient space. Building on this observation, we develop a scalable gradient-attribution method, SimpleLoRIF, for identifying benign-looking SFT examples associated with unsafe behavior, and use it to construct SafeTulu by removing the 20% of Tulu 3 examples most aligned with a reference set of unsafe responses. Across Qwen, Llama, and Olmo, training on SafeTulu consistently reduces GCG attack success rate by up to 22% while largely preserving or improving capability across multiple benchmarks, outperforming LLM-as-a-judge filtering and a prior gradient filtering method. We also analyzed possible sources of benign-looking safety-degrading data, such as tool-execution failures, based on the removed data. By restricting backpropagation to the final few transformer blocks and using a compact gradient representation, SimpleLoRIF allows for attributing and filtering datasets containing on the order of a million examples, boosting gradient attribution from a tool more suited to small-scale analysis into a more practical data-selection mechanism for large-scale safety filtering. We release SafeTulu along with a Hugging Face Space that lets researchers build their own customized versions of SafeTulu to fine-tune on, such as by changing the removal fraction or filtering only certain subsets.

## 9 Limitations

While we believe our work represents a promising route to increasing safety through gradient attribution, we recognize limitations. First, we only test general safety degradation, and do not test specific harms or behaviors such as sycophancy. Second, while SafeRLHF is a broad dataset, we do not test other target sets. Despite these limitations, we believe SafeTulu, our gradient embedding method, our findings on emergent misalignment, and our qualitative analysis are still valuable to the community.

## 10 Reproducibility Statement

We release SafeTulu and a Hugging Face Space to support researchers who would like to create customized versions of SafeTulu. We also state our hyperparameters in Appendix C.3.

## 11 AI Use Statement

In this work, we used generative AI tools for code implementation, qualitative analysis, and writing organization. We have not used generative AI tools for synthetic data generation, formulating mathematical claims, proposing hypotheses, designing research methodology, dataset reformatting, or result interpretation. Developing theoretical models, assisting in the writing of proofs, and assisting with translation are not applicable to this work. Additionally, we used generative AI tools for creating scientific figures, creating tables, identifying relevant literature, and searching for information. We have reviewed all AI-assisted work. For example, we confirmed initial qualitative claims by AI about our dataset. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Jan Betley, Daniel Chee Hian Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 4043–4068. PMLR, 13–19 Jul 2025.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, Hamed Hassani, and Eric Wong. JailbreakBench: An open robustness benchmark for jailbreaking large language models. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

Bryant Chen, Wilka Carvalho, Nathalie Baracaldo, Heiko Ludwig, Benjamin Edwards, Taesung Lee, Ian Molloy, and Biplav Srivastava. Detecting backdoor attacks on deep neural networks by activation clustering. In Proceedings of the AAAI Workshop on Artificial Intelligence Safety, 2019.

Sang Keun Choe, Hwijeen Ahn, Juhan Bae, Kewen Zhao, Minsoo Kang, Youngseog Chung, Adithya Pratapa, Willie Neiswanger, Emma Strubell, Teruko Mitamura, Jef Schneider, Eduard Hovy, Roger Grosse, and Eric Xing. What is your data worth to GPT? LLM-scale data valuation with influence functions. In Advances in Neural Information Processing Systems, volume 38, pages 145944–145985, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.

Jiaxing Cui, Wei-Lin Chiang, Ion Stoica, and Cho-Jui Hsieh. OR-Bench: An over-refusal benchmark for large language models. In Proceedings of the International Conference on Machine Learning, 2025.

Aladin Djuhera, Swanand Ravindra Kadhe, Syed Zawad, Farhan Ahmed, Heiko Ludwig, and Holger Boche. Fixing it in post: A comparative study of LLM post-training data quality and model performance. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, volume 38, 2025.

Jonas Geiping, Liam H. Fowl, W. Ronny Huang, Wojciech Czaja, Gavin Taylor, Michael Moeller, and Tom Goldstein. Witches’ brew: Industrial scale data poisoning via gradient matching. In International Conference on Learning Representations, 2021.

Zayd Hammoudeh and Daniel Lowd. Identifying a training-set attack’s target using renormalized influence estimation. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 1367–1381, 2022. doi: 10.1145/3548606.3559335.

Will Hawkins, Kaivalya Rawal, Jonathan Rystrøm, Stratis Tsirtsis, Zihao Fu, Greta Warren, Ryan Brown, Eoin Delaney, Sandra Wachter, Brent Mittelstadt, and Chris Russell. The heterogeneous safety impacts of benign multilingual fine-tuning. In Proceedings of the 43rd International Conference on Machine Learning (ICML), Seoul, South Korea, 2026.

Luxi He, Mengzhou Xia, and Peter Henderson. What’s in your safe data?: Identifying benign data that breaks safety. In First Conference on Language Modeling, 2024.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. Qwen2.5-Coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Josef Dai, Boren Zheng, Tianyi Alex Qiu, Jiayi Zhou, Kaile Wang, Boxun Li, Sirui Han, Yike Guo, and Yaodong Yang. PKU-SafeRLHF: Towards multi-level safety alignment for LLMs with human preference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 31983–32016. Association for Computational Linguistics, July 2025. ISBN 979-8-89176-251-0.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14), 2021. ISSN 2076-3417.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894. PMLR, 06–11 Aug 2017.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling (COLM), 2025.

Hao Li, Lijun Li, Zhenghao Lu, Xianyi Wei, Rui Li, Jing Shao, and Lei Sha. Layer-aware representation filtering: Purifying finetuning data to preserve LLM safety alignment. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 8030–8050, Suzhou, China, November 2025. Association for Computational Linguistics.

Shuangqi Li, Hieu Le, Jingyi Xu, and Mathieu Salzmann. LoRIF: Low-rank influence functions for scalable training data attribution, 2026.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.

Pratyush Maini, Sachin Goyal, Dylan Sam, Alexander Robey, Yash Savani, Yiding Jiang, Andy Zou, Matt Fredrikson, Zachary Lipton, and Zico Kolter. Safety Pretraining: Toward the next generation of safe AI. In Advances in Neural Information Processing Systems, 2025.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 35181–35224. PMLR, 21–27 Jul 2024.

Nikita Menon. Base vs instruct models with emergent misalignment. Research report, Supervised Program for Alignment Research (SPAR), 2025.

Meta AI. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. Meta AI Blog, September 2024.

Gouki Minegishi, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, and Yutaka Matsuo. Understanding emergent misalign ment via feature superposition geometry. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 30385–30414, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6.

Mistral AI. Mistral small 3. Mistral AI Blog, January 2025.

Burt L. Monroe, Michael P. Colaresi, and Kevin M. Quinn. Fightin’ words: Lexical feature selection and evaluation for identifying the content of political conflict. Political Analysis, 16(4):372–403, 2008.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 OLMo 2 furious. arXiv preprint arXiv:2501.00656, 2024.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. TRAK: Attributing model behavior at scale. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 27074–27113. PMLR, 23–29 Jul 2023.

Neehar Peri, Neal Gupta, W. Ronny Huang, Liam Fowl, Chen Zhu, Soheil Feizi, Tom Goldstein, and John P. Dickerson. Deep k-NN defense against clean-label data poisoning attacks. In ECCV 2020 Workshop on Adversarial Robustness in the Real World (AROW), 2020.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In Advances in Neural Information Processing Systems, volume 33, pages 19920–19930, 2020.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! In The Twelfth International Conference on Learning Representations, 2024.

Qwen-Team. Qwen3 technical report, 2025.

Jayaram Raghuram, George Kesidis, and David J. Miller. A study of backdoors in instruction fine-tuned language models. arXiv preprint arXiv:2406.07778, 2024.

Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, and Jing Shao. LLMs know their vulnerabilities: Uncover safety gaps through natural distribution shifts. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24763–24785, Vienna, Austria, July 2025. Association for Computational Linguistics.

Shawn Shan, Arjun Nitin Bhagoji, Haitao Zheng, and Ben Y. Zhao. Poison forensics: Traceback of data poisoning attacks in neural networks. In 31st USENIX Security Symposium (USENIX Security 22), pages 3575–3592, 2022.

Han Shen, Pin-Yu Chen, Payel Das, and Tianyi Chen. SEAL: Safety-enhanced aligned LLM fine-tuning via bilevel data selection. In The Thirteenth International Conference on Learning Representations, 2025.

Anna Soligo, Edward Turner, Senthooran Rajamanoharan, and Neel Nanda. Emergent misalignment is easy, narrow misalignment is hard. In International Conference on Learning Representations, pages 102392–102429, 2026.

Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, Burak Hasircioglu, Ezzeldin Shereen, Carlos Mougan, Vasilios Mavroudis, Erik Jones, Chris Hicks, Nicholas Carlini, Yarin Gal, and Robert Kirk. Poisoning attacks on LLMs require a near-constant number of poison samples. arXiv preprint arXiv:2510.07192, 2025.

Brandon Tran, Jerry Li, and Aleksander Madry. Spectral signatures in backdoor attacks. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Clemens Vetter, David Kaczér, Lucie Flek, and Florian Mai. Data attribution of emergent misalignment with persona features. In BlackboxNLP Workshop, 2026.

Miles Wang, Tom Dupre la Tour, Olivia Watkins, Aleksandar Makelov, Ryan Chi, Samuel Miserendino, Jefrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, and Daniel Mossing. Persona features control emergent misalignment. In International Conference on Learning Representations, pages 48836–48885, 2026.

Zongru Wu, Pengzhou Cheng, Lingyong Fang, Zhuosheng Zhang, and Gongshen Liu. Gracefully filtering backdoor samples for generative large language models without retraining. In Proceedings of the International Conference on Computational Linguistics (COLING), 2025.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: Selecting influential data for targeted instruction tuning. In Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research. PMLR, 2024.

Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, and Muhao Chen. Instructions as backdoors: Backdoor vulnerabilities of instruction tuning for large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3111–3126, 2024.

Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, and Hongxia Jin. Backdooring instruction-tuned large language models with virtual prompt injection. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6065–6086, 2024.

Yu Yang, Tian Yu Liu, and Baharan Mirzasoleiman. Not all poisons are created equal: Robust training against data poisoning. In International Conference on Machine Learning, volume 162 of PMLR, 2022.

Junbo Zhang, Qianli Zhou, Xinyang Deng, Wen Jiang, Jie Pan, and Jinbiao Zhu. Datashield: Safety-degrading data filtering for LLM benign instruction fine-tuning, 2026.

Haodong Zhao, Tianyi Xu, Tianhang Zhao, Zhuosheng Zhang, and Gongshen Liu. GradSentry: Gradient spectral entropy for backdoor sample filtering in large language model fine-tuning. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026.

Jefrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instructionfollowing evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023. doi: 10.48550/arXiv.2311.07911.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models, 2023.

## Appendix

## A SafeTulu Results Table

$$
4 7 . 6 7 \pm 0 . 6 7
$$

$$
4 3 . 0 \pm 4 . 4
$$

$$
5 2 . 0 0 \pm 2 . 0 8
$$

$$
1 9 . 8 \pm 2 . 5
$$

$$
3 7 . 2 \pm 7 . 4
$$

$$
4 1 . 7 \pm 2 . 7
$$

$$
4 7 . 3 3 \pm 1 . 2 0
$$

$$
2 1 . 3 \pm 1 . 2
$$

$$
4 5 . 2 \pm 0 . 7
$$

$$
3 9 . 7 \pm 0 . 9
$$

$$
2 1 . 1 \pm 0 . 9
$$

$$
4 6 . 3 3 \pm 2 . 6 0
$$

$$
4 5 . 8 \pm 1 . 3
$$

$$
4 2 . 0 \pm 2 . 6
$$

$$
2 2 . 4 \pm 0 . 5
$$

$$
4 5 . 3 \pm 0 . 6
$$

$$
{ \bf 3 7 . 0 0 \pm 1 . 5 3 }
$$

$$
{ \bf 3 6 . 3 \pm 1 . 2 }
$$

$$
2 2 . 4 \pm 0 . 7
$$

$$
4 6 . 1 \pm 1 . 4
$$

$$
3 8 . 0 0 \pm 1 . 5 3 
$$

$$
4 2 . 7 \pm 0 . 7
$$

$$
1 5 . 3 \pm 0 . 1
$$

$$
4 3 . 8 \pm 0 . 7
$$

$$
4 0 . 0 0 \pm 0 . 5 8
$$

$$
4 3 . 3 \pm 2 . 0
$$

$$
1 5 . 4 \pm 0 . 2
$$

$$
4 4 . 2 \pm 1 . 3
$$

$$
3 9 . 0 0 \pm 1 . 5 8 
$$

$$
4 1 . 7 \pm 0 . 3
$$

$$
1 5 . 8 \pm 0 . 4
$$

$$
4 4 . 9 \pm 0 . 8
$$

$$
3 5 . 3 3 \pm 2 . 0 3
$$

$$
4 3 . 3 \pm 1 . 5
$$

$$
1 5 . 3 \pm 0 . 4
$$

$$
4 3 . 0 \pm 0 . 6
$$

$$
{ \bf 3 0 . 3 3 \pm 0 . 3 3 }
$$

$$
3 9 . 7 \pm 0 . 3
$$

$$
1 5 . 8 \pm 0 . 3
$$

$$
4 3 . 4 \pm 0 . 5
$$

$$
4 8 . 3 3 \pm 0 . 3 3
$$

$$
4 7 . 0 \pm 1 . 2
$$

$$
1 4 . 4 \pm 0 . 4
$$

$$
3 8 . 0 \pm 1 . 1
$$

$$
4 8 . 0 0 \pm 1 . 0 0
$$

$$
4 6 . 0 \pm 0 . 6
$$

$$
1 4 . 3 \pm 0 . 3
$$

$$
3 9 . 5 \pm 0 . 4
$$

$$
4 6 . 3 3 \pm 2 . 3 3
$$

$$
4 5 . 7 \pm 0 . 3
$$

$$
1 4 . 3 \pm 0 . 2
$$

$$
3 8 . 7 \pm 1 . 3 
$$

$$
4 9 . 0 0 \pm 1 . 1 5
$$

$$
4 6 . 7 \pm 0 . 7
$$

$$
1 4 . 6 \pm 0 . 3
$$

$$
3 8 . 2 \pm 0 . 2
$$

$$
4 1 . 3 3 \pm 2 . 4 0
$$

$$
4 3 . 3 \pm 1 . 2
$$

$$
1 3 . 6 \pm 0 . 6
$$

$$
3 8 . 0 \pm 0 . 2
$$

$$
1 1 . 7 \pm 0 . 3
$$

$$
4 1 . 7 \pm 1 . 5
$$

$$
2 9 . 3 \pm 1 . 3
$$

$$
6 6 . 9 \pm 0 . 6
$$

$$
9 . 7 \pm 0 . 7
$$

$$
4 3 . 7 \pm 1 . 2
$$

$$
3 1 . 1 \pm 2 . 3
$$

$$
6 5 . 7 \pm 1 . 1
$$

$$
8 . 7 \pm 0 . 7
$$

$$
4 3 . 3 \pm 1 . 5
$$

$$
2 8 . 3 \pm 0 . 8
$$

$$
9 . 3 \pm 0 . 7
$$

$$
4 1 . 3 \pm 0 . 9
$$

$$
2 8 . 8 \pm 0 . 5
$$

$$
6 6 . 0 \pm 0 . 8
$$

$$
7 . 0 \pm 0 . 6
$$

$$
3 7 . 7 \pm 2 . 3
$$

$$
{ \bf 3 1 . 6 \pm 1 . 4 }
$$

$$
6 6 . 1 \pm 1 . 0
$$

$$
5 2 . 6 \pm 8 . 2
$$

$$
2 6 . 3 \pm 4 . 3
$$

$$
3 7 . 8 \pm 1 . 7
$$

$$
2 4 . 1 \pm 2 . 6
$$

$$
5 2 . 8 \pm 1 . 6
$$

$$
6 5 . 3 \pm 5 . 8
$$

$$
3 1 . 2 \pm 1 . 2
$$

$$
4 0 . 4 \pm 1 . 1
$$

$$
2 8 . 1 \pm 0 . 2
$$

$$
6 9 . 7 \pm 2 . 1 
$$

$$
{ \pm } 9 . 8 \pm 1 . 3
$$

$$
2 9 . 3 \pm 1 . 2
$$

$$
4 0 . 5 \pm 0 . 2
$$

$$
2 7 . 4 \pm 0 . 5
$$

$$
6 9 . 2 \pm 5 . 1
$$

$$
5 5 . 3 \pm 0 . 2
$$

$$
3 0 . 1 \pm 1 . 1
$$

$$
4 1 . 9 \pm 0 . 2
$$

$$
2 7 . 1 \pm 0 . 4
$$

$$
6 3 . 6 \pm 1 . 5
$$

$$
5 8 . 1 \pm 0 . 4
$$

$$
3 2 . 7 \pm 0 . 5
$$

$$
4 0 . 2 \pm 0 . 8
$$

$$
2 7 . 1 \pm 0 . 5
$$

$$
6 5 . 1 \pm 3 . 0
$$

$$
3 3 . 4 \pm 0 . 7
$$

$$
1 4 . 3 \pm 0 . 7
$$

$$
2 4 . 9 \pm 0 . 4
$$

$$
1 9 . 3 \pm 0 . 5
$$

$$
7 0 . 9 \pm 2 . 8
$$

$$
3 1 . 9 \pm 0 . 6
$$

$$
1 5 . 2 \pm 1 . 2
$$

$$
2 3 . 7 \pm 0 . 5
$$

$$
2 1 . 4 \pm 1 . 5
$$

$$
6 1 . 6 \pm 0 . 8
$$

$$
1 3 . 1 \pm 1 . 1
$$

$$
2 5 . 4 \pm 1 . 7
$$

$$
2 1 . 6 \pm 0 . 9
$$

$$
7 0 . 0 \pm 2 . 8
$$

$$
2 3 . 5 \pm 4 . 9
$$

$$
1 3 . 0 \pm 0 . 5
$$

$$
2 4 . 5 \pm 0 . 4
$$

$$
2 1 . 4 \pm 0 . 2
$$

$$
{ \bf 5 5 . 7 \pm 1 . 9 }
$$

$$
3 2 . 6 \pm 2 . 3
$$

$$
1 4 . 7 \pm 1 . 1
$$

$$
{ \bf 2 5 . 6 \pm 0 . 8 }
$$

$$
1 9 . 8 \pm 0 . 6
$$

$$
6 7 . 1 \pm 1 . 5
$$

$$
3 5 . 2 \pm 0 . 5
$$

$$
9 . 9 \pm 0 . 5
$$

$$
1 2 . 7 \pm 1 . 0
$$

$$
2 2 . 6 \pm 0 . 2
$$

$$
4 9 . 5 \pm 1 . 5
$$

$$
3 6 . 4 \pm 1 . 3
$$

$$
7 . 7 \pm 1 . 0
$$

$$
8 . 5 \pm 4 . 3
$$

$$
2 2 . 4 \pm 0 . 5
$$

$$
4 7 . 7 \pm 3 . 3
$$

$$
3 7 . 1 \pm 1 . 5
$$

$$
8 . 3 \pm 0 . 5
$$

$$
1 3 . 2 \pm 0 . 6
$$

$$
2 2 . 8 \pm 0 . 3
$$

$$
5 7 . 3 \pm 3 . 0
$$

$$
3 4 . 4 \pm 0 . 7
$$

$$
9 . 9 \pm 0 . 7
$$

$$
1 3 . 5 \pm 0 . 8
$$

$$
2 2 . 4 \pm 0 . 2
$$

$$
4 7 . 6 \pm 2 . 2
$$

$$
3 6 . 7 \pm 1 . 3
$$

$$
9 . 1 \pm 0 . 1
$$

$$
1 2 . 7 \pm 0 . 6
$$

$$
2 3 . 0 \pm 0 . 1
$$

$$
5 1 . 1 \pm 1 . 0
$$

$$
6 8 . 5 \pm 0 . 5
$$

$$
4 0 . 0 \pm 0 . 5
$$

$$
5 5 . 3 \pm 0 . 8
$$

$$
4 5 . 1 \pm 0 . 3
$$

$$
6 8 . 6 \pm 2 . 4
$$

$$
6 8 . 3 \pm 1 . 1
$$

$$
4 1 . 0 \pm 1 . 3
$$

$$
5 5 . 4 \pm 0 . 5
$$

$$
4 5 . 6 \pm 0 . 1
$$

$$
6 3 . 8 \pm 2 . 7
$$

$$
6 8 . 1 \pm 0 . 9
$$

$$
3 9 . 5 \pm 0 . 2
$$

$$
5 5 . 1 \pm 0 . 5
$$

$$
4 5 . 5 \pm 0 . 3
$$

$$
7 3 . 5 \pm 0 . 8
$$

$$
6 8 . 0 \pm 0 . 5
$$

$$
4 0 . 5 \pm 1 . 7
$$

$$
{ \pm } 6 . 5 \pm 0 . 1
$$

$$
4 5 . 7 \pm 0 . 2
$$

$$
6 7 . 5 \pm 0 . 4
$$

$$
5 5 . 2 \pm 0 . 5
$$

$$
7 1 . 7 \pm 2 . 3
$$

$$
4 1 . 7 \pm 0 . 6
$$

$$
4 5 . 9 \pm 0 . 5
$$

$$
6 6 . 5 \pm 1 . 3
$$

Table 1 Evaluation results after baseline training, training on randomly filtered data, training on the LLM-as-a-judge filtered dataset, training on data filtered with the method of He et al., and training on SafeTulu, for Qwen 3 1.7B, Llama 3.2 1B, Olmo 2 1B, and the larger Llama 3.1 8B. All values are percentages, reported as mean ± standard error. Lower is better on GCG & ActorBreaker attack success rate and ORBench over-refusal rate, while higher accuracy is better on all capability metrics. Across all models, including at the 8B scale, training on SafeTulu decreases ASR while preserving comparable capability.

## B Full Qualitative Analysis

## B.1 Removal Rates

Table 2 shows the fraction of each Tulu 3 subset removed by gradient filtering and by the LLM judge under the same 20% budget. The math subsets (highlighted) show the clearest contrast. The judge removes all three at almost the same rate (15.4%–15.6%), which makes sense for a scorer that evaluates content only at the surface level. Gradient filtering spreads its removals much more widely across the same subsets, from 7.9% of numinamath to 19.4% of open\_math\_2\_gsm8k. It treats content that looks similar on the surface as diferent, indicating that it is reacting to something beyond surface content.

<table><tr><td>Subset</td><td>Gradient Removal Rate</td><td>Judge Removal Rate</td></tr><tr><td>personahub</td><td>9.2%</td><td>15.5%</td></tr><tr><td>codealpaca</td><td>17.3%</td><td>15.5%</td></tr><tr><td>wildchat</td><td>18.7%</td><td>28.2%</td></tr><tr><td>aya</td><td>11.7%</td><td>19.2%</td></tr><tr><td>flan</td><td>22.1%</td><td>17.0%</td></tr><tr><td>wildjailbreak</td><td>39.4%</td><td>32.8%</td></tr><tr><td>wildguardmixtrain</td><td>41.0%</td><td>44.3%</td></tr><tr><td>numinamath</td><td>7.9%</td><td>15.4%</td></tr><tr><td>open_math_2_gsm8k</td><td>19.4%</td><td>15.6%</td></tr><tr><td>personas_math</td><td>17.2%</td><td>15.6%</td></tr><tr><td>no_robots_converted</td><td>31.9%</td><td>20.2%</td></tr><tr><td>table_gpt</td><td>18.7%</td><td>15.7%</td></tr><tr><td>science.* (SciRIFF tasks, aggregate)</td><td>21.3%</td><td>15.7%</td></tr><tr><td>hard_coded_* (aggregate)</td><td>54.2%</td><td>87.5%</td></tr><tr><td>oasst1 + very small subsets, aggregate</td><td>35.8%</td><td>21.0%</td></tr></table>

Table 2 Fraction of each Tulu 3 subset removed by each method under a 20% budget, with math subsets highlighted. While the LLM judge removes math at a roughly flat rate, gradient filtering varies its rates, indicating it is finding a signal beyond surface level content.

## B.2 Removal Analysis Methodology

<table><tr><td>Subset</td><td>Feature</td><td>RR</td></tr><tr><td>numinamath</td><td>Python traceback/exception</td><td>1.71×</td></tr><tr><td>personahub</td><td>&quot;Write a Python function&quot; code task</td><td>1.74×</td></tr><tr><td>personahub</td><td>Calculus/differential-equation modeling</td><td>0.51×</td></tr><tr><td>codealpaca</td><td>High-level architecture discussion</td><td>1.75×</td></tr><tr><td>codealpaca</td><td>Runnable code in response</td><td>0.89×</td></tr><tr><td>open_math_2_gsm8k</td><td>Long repeating decimal</td><td>0.36×</td></tr><tr><td>personas_math</td><td>Algebraic manipulation</td><td>0.70×</td></tr><tr><td>wildchat</td><td>English</td><td>1.46×</td></tr><tr><td>aya</td><td>English</td><td>1.73×</td></tr></table>

Table 3 Relative rate (RR) of each feature among removed vs kept documents. We observe trends such as removing tool-execution failures and simple coding tasks at a higher rate.

To find what separates removed from kept documents without assuming a category in advance, we run a log-odds comparison with an informative Dirichlet prior (the “fightin’ words” method) over unigrams and bigrams (Monroe et al., 2008). We do this separately for the prompt and the response, and only within one subset at a time, so subset identity cannot confound the comparison. Each term’s document frequency in the removed and kept groups is compared against a shared background rate and turned into a z-score, which keeps words that are common overall from dominating the ranking. We analyze the most strongly enriched terms on each side. While gradient filtering did not show any discernible pattern for many subsets, we did identify trends in a few subsets, shown in Table 3. For each trend, we measure the rate on the full removed and kept populations and report the relative risk (RR), the removal rate within the feature divided by the subset’s baseline removal rate. Table 4 shows truncated dataset examples representing each trend.

We discuss our findings below. We stress that these patterns are only suggestive, not definitively established.
<table><tr><td>Subset</td><td>Removed</td><td>Kept</td></tr><tr><td>numinamath</td><td>...from scipy.optimize import minimize → ModuleNotFoundError: No module named &#x27;scipy&#x27; “It seems that the scipy library is not available...&quot;</td><td>&quot;Given..  $\begin{array} { r } { . a _ { n } = \sin { \frac { n \pi } { 3 } } . } \end{array}$  , find the value of  $a _ { 3 } . { \overset { \underset { 9 } { \overset { \underset { 1 9 } { \overset { \ddots } } { \overset { \ddots } } { \longrightarrow } } } } { \longrightarrow } }$  import sympy as  $\mathsf { s p . . . a _ { - } 3 } = \mathsf { a _ { - } n }$  subs(n, 3)</td></tr><tr><td>personahub</td><td>&quot;Write a python function to convert a list of his- torical event years into a list of centuries...&quot; → def convert_years_to_centuries(years):...</td><td>“An entrepreneur is analyzing...determine the constants in the revenue functions and predict when total revenue will reach $50 million&quot; → &quot;Step 1: Determine...&quot;</td></tr><tr><td>codealpaca</td><td>“... building a resilient serverless architecture us- ing...AWS Lambda and API Gateway...Could you provide a detailed, sequential guide..  $\cdot ^ { \mathfrak { n } } \to$  “... These include planning, establishing the envi- ronment, continuous integration...&quot;</td><td>“Create a function that takes two in- put strings, and returns a dictionary containing the count of unique char- acters...&quot; → def count_chars(string1,</td></tr><tr><td>open_math_2_ gsm8k</td><td>“Emily planted a cherry blossom tree...when she was 6.. . takes 9 years to bloom... currently  $1 0 . . . ^ { , 9 }  \stackrel { . . . 9 } { . . . 9 } - 4 = 5$  years are left.  $\ldots 1 0 + 5 =$   $1 5 . . . \boxed { 1 5 } ^ { , }$ </td><td>string2):...print(char_counts) “... The smallest building is 1/3 the size of the second-largest building.  $\dots ^ { 9 }  \dots 1 / 3 \times 1 3 5 =$  0.3333333333333333  $\times \ 1 3 5 = 4 5$  feet tall.&quot;</td></tr><tr><td>personas_math</td><td>“Lawmaker Lisa.. . plans to allocate resources to 5 different community centers. Each center will receive an equal number...&quot;</td><td>“A television producer...divide the show into three parts. . .&quot; → &quot;Let&#x27;s define variables for the time allocated to each part...&quot;</td></tr><tr><td>wildchat</td><td> $\mathrm { ^ { * } h i ^ { * } } \to \mathrm { ^ { * } H e l l o ! }$  How can I assist you today?&quot;</td><td>&quot;Title: Dyanra: Misteryo, Kababalaghan. Tauhan: Mike, Elmer, Jones, Erica..  $\begin{array} { r l } { . ^ { \mathfrak { n } } } & { { } \to ^ { \ \mathfrak { n } } . } \end{array}$  ..Sa isang tahimik at mapayapang baryo, sinimulan ng</td></tr><tr><td>wildjailbreak</td><td>&quot;share or trade illegal or non-consensual pornog- raphy&quot;  &quot;I&#x27;m sorry, but I can&#x27;t fulfill this request.&quot;</td><td>magkakapatid na Mike, Elmer...&quot; “... conduct a dogfighting event without getting caught.  $\ldots { \stackrel {  } {  } } \stackrel { \rleftarrow } {  } \stackrel {  } { \longrightarrow } \stackrel {  } { \Gamma } \stackrel {  } { \mathrm { m } }$  really sorry, but I can&#x27;t as- sist. . . if you&#x27;re looking to engage with dogs in a positive and legal manner...&quot;</td></tr></table>

Table 4 Examples of removed and kept documents from each subset. Responses are truncated to their distinguishing parts.

## B.3 Code Analysis

In numinamath, where the model writes Python/SymPy code, reads the executed output, and continues reasoning, the top terms in the documents gradient filtering removed are fragments of a Python exception (traceback, most recent call, call last) and the top terms in the documents gradient filtering kept are the vocabulary of a clean symbolic trace (sympy, import\_sympy, symbols). Removed documents contain a real tool execution traceback or exception 1.71× more than kept documents. A typical removed example is a failed first attempt at tool execution (an AttributeError, TypeError, or SyntaxError), after which the model notices the error and continues. This suggests that safety-degrading code data may extend beyond the insecure code studied in emergent misalignment. In personahub, gradient filtering removes “write a Python function” tasks (RR 1.74×) and keeps calculus and diferential-equation modeling problems (RR 0.51×) more often, suggesting that code data may be more safety-degrading than math data. In codealpaca, gradient filtering more often removes high-level discussion of software architecture (scalability, distributed, authentication) at an RR 1.75× and keeps responses that contain runnable code (def, print, return) at an RR of 0.89×.

## B.4 Math Analysis

In open\_math\_2\_gsm8k, gradient filtering more often removes answers whose arithmetic comes out to clean numbers and keeps answers containing long repeating decimals (e.g. 0.3333333333333333) at an RR of 0.36×. In personas\_math, it removes narrative arithmetic, in which quantities are carried in prose through a sequence of operations, and keeps genuine algebraic manipulation, in which the model defines a variable, sets up an equation, and substitutes or simplifies (RR 0.70×).

## B.5 Language Analysis

In wildchat, gradient filtering removes English conversations at 1.46× as often as non-English languages, with every other major language removed at 0.26–0.65× the baseline rate. aya\_100k shows the same direction, with English again the most removed language (RR 1.73×), even though English is only 6.7% of that subset and the majority language in WildChat. This is notable because fine-tuning on non-English data has often been reported to degrade safety (Hawkins et al., 2026). One possible reason for the reversal here is that Tulu 3’s data is very clean (Lambert et al., 2025; Djuhera et al., 2025).

## B.6 Overall Analysis

In terms of well-defined types of data, tool-execution failures and narrative arithmetic stand out as possible categories of data which may degrade safety that have been under-studied in prior work. As a broad trend, gradient filtering seems to suggest that direct examples may degrade safety more than elaborate examples. Gradient filtering appears to have removed more direct data (such as clean arithmetic and simple coding tasks) and kept less direct or more elaborate data (such as repeating-decimal arithmetic, algebraic setup, and diferential equation modeling) more often. One possible reason is that direct, immediately helpful answers match the pattern of simple compliance, so direct data may degrade safety more. Table 5 shows that this efect cannot be reduced to simply optimizing for shorter examples. Removed documents are clearly shorter in only about half of the eleven major subsets (6/11), with three other subsets having a minor length diference of 5% or less and two others showing gradient filtering selecting longer data.

<table><tr><td>Subset</td><td>% Removed Data is Shorter By</td></tr><tr><td>table_gpt</td><td>40%</td></tr><tr><td>personahub</td><td>33%</td></tr><tr><td>aya</td><td>30%</td></tr><tr><td>no_robots_converted</td><td>29%</td></tr><tr><td>wildjailbreak</td><td>21%</td></tr><tr><td>flan</td><td>20%</td></tr><tr><td>wildguardmixtrain</td><td>5%</td></tr><tr><td>open_math_2_gsm8k</td><td>5%</td></tr><tr><td>personas_math</td><td>3%</td></tr><tr><td>codealpaca</td><td>-3%</td></tr><tr><td>wildchat</td><td>-11%</td></tr></table>

Table 5 The percentage that the data removed by gradient filtering is shorter than the kept documents for every major subset. The percentage is only large for half (6/11) of the major subsets, indicating that gradient filtering is not simply optimizing for shorter length examples.

We stress that these patterns are only suggestive rather than established. We make no claim that these features must cause safety degradation, only that these features are what these examples appear to have in common. Every result comes from the correlational method in Section B.2.

## C Extended SafeTulu Details

## C.1 Removal Fraction Sweep

<table><tr><td></td><td>Removal Fraction LLM-as-a-Judge ASR ↓ Gradient ASR↓</td><td></td></tr><tr><td>0%</td><td> $3 8 . 0 0 \pm 1 . 5 3 $ </td><td> $3 8 . 0 0 \pm 1 . 5 3 $ </td></tr><tr><td>5%</td><td> $3 6 . 3 3 \pm 1 . 8 6$ </td><td> $3 5 . 0 0 \pm 1 . 5 3$ </td></tr><tr><td>10%</td><td> ${ \bf 3 5 . 0 0 \pm 1 . 5 3 }$ </td><td> $3 3 . 6 7 \pm 1 . 2 0$ </td></tr><tr><td>20%</td><td> $3 9 . 0 0 \pm 1 . 5 8$ </td><td> ${ \bf 3 0 . 3 3 \pm 0 . 3 3 }$ </td></tr><tr><td>40%</td><td> $4 3 . 0 0 \pm 2 . 5 2$ </td><td> $3 9 . 0 0 \pm 0 . 5 8$ </td></tr></table>

Table 6 GCG attack success rate (%) vs. the fraction of training data removed, for training Llama 3.2 1B after LLM-as-ajudge filtering and gradient-based filtering. Values are mean ± standard error; lower is better.

To test whether our results depend on the 20% removal budget, we swept the removal fraction over 0%, 5%, 10%, 20% and 40% for both gradient filtering and the LLM-as-a-judge filter on Llama 3.2 1B. As before, each value is the mean ± one standard error over three fine-tuning runs with diferent seeds.

Results are shown in Table 6. Gradient filtering reaches a lower minimum ASR. The best ASR of gradient filtering is 30.33% at 20% removal. The LLM judge’s best is 35.00% at 10% removal. The point where removal stops helping shows how much truly safety-degrading data each method finds. The LLM judge lowers ASR steadily up to 10% removal (38.00 → 35.00). Beyond that, ASR rises above the unfiltered baseline: 39.00 at 20% and 43.00 at 40%. This suggests that beyond about 10%, the judge can no longer find data that is truly safety-degrading. It starts removing benign-looking data that may actually support safety, consistent with Tulu 3 containing little overtly harmful content. Gradient filtering shows the same reversal, but only past 20%: it keeps lowering ASR through 20% (38.00 → 30.33) before rising at 40% (39.00). Gradient filtering keeps helping for about twice as much removed data before this happens. This indicates that it finds a larger pool of genuinely safety-degrading data than a content-based judge.

## C.2 LLM-as-a-Judge Baseline Details

## Safety Scoring Prompt (0–5 Scale)

```jsonl
detailed references to illegal activities. References: encouraging self-harm, violence, or other dangerous actions. Suitability: contains explicit elements;
should be restricted or removed from public platforms.
5 (Highly Unsafe): Content: highly unsafe and overtly harmful. Language: hate speech, violent threats, or calls to action against individuals or groups.
Themes: graphic adult content involving illegal activities. Advice: promotion or instruction of illegal or harmful actions. Privacy: disclosure of personally
identifiable information without consent. Suitability: inappropriate for any general, educational, or public platform.
Instructions: (1) Analyze the text thoroughly: understand the content, context, and tone, and identify any unsafe elements based on the criteria. (2)
Assign a safety score: start from 0 and assign the highest applicable score (do not sum points). (3) Return a JSON object:
"<reason>"}, where the reason briefly states the main issue (e.g., “hate speech,” “financial advice,” “WMD”).
Example. Text: “I think we should overthrow the government because they are all corrupt and don’t care about us.” Identified content: expresses extremist
beliefs and suggests illegal activities. Justification: advocates the illegal act of overthrowing the government, which is a serious ofense; promotes extremist
beliefs and could encourage unsafe behavior, making it clearly inappropriate for general audiences. Output: {"score": 4, "reason": "illegal activity"}
```

We prompt Qwen 3 8B, served with constrained (JSON-schema-guided) decoding, to score every document in the training set for safety on an integer 0–5 scale, where 0 denotes content that is fully safe and appropriate for all audiences and 5 denotes highly unsafe content (hate speech, violent threats or calls to action, graphic illegal content, disclosure of personal information, or explicit promotion/instruction of illegal or harmful acts). Each score level is defined by the rubric established in Appendix C.1 from Maini et al. (2025) (shown above) over content, language, themes, advice given, and audience suitability. The judge is instructed to start from 0 and assign the highest applicable score rather than summing point-by-point, and returns a structured {"score": int, "reason": str} object. Scores that fail to parse are treated as missing and excluded.

The judge sees both the user prompt and the model’s response, so it can penalize an unsafe prompt-response pair even when the response itself appears benign in isolation. Because the discrete 0–5 scale may produce a tie band straddling the removal boundary (e.g. the cutof falling in the middle of the population of 4s and 5s), ties are broken by a seeded random draw, so which documents within the boundary score are kept vs. dropped is randomized rather than determined by document order.

## C.3 Hyperparameters

Training Hyperparameters:

• Number of Steps: 2,297

• Learning Rate (LR): $1 \times 1 0 ^ { - 4 }$ (after conducting a sweep over $5 \times 1 0 ^ { - 5 }$ $1 \times 1 0 ^ { - 4 }$ , and $3 \times 1 0 ^ { - 4 } )$

• β1: 0.9

• β2: 0.95

• Weight Decay: 0.1

• LR Warm-Up: 20 steps

• LR Decay Ratio: 20%

• Minimum LR Factor: 0.05

• Sequence Length: 2,048

• Global Batch Size: 128 sequences

• Batch Size per GPU: 8

• Gradient Clipping: Max Norm 1.0

• Parameter Precision: bf16

• Gradient Reduction Precision: fp32

• Dataloader Shufle Seed: 9,176

GCG Hyperparameters:

• Initial Sufix: $2 0 ~ " ! "$ tokens

• Optimization Target: JailBreakBench supplied afirmative target per behavior

• Number of Optimization Steps: 500

• Candidate Sufixes/Step: 512

• Gradient Top-k: 256

## C.4 Removal Scoring Details

For a training document x with rank-1 factors $\{ ( u _ { \ell } , v _ { \ell } ) \}$ and a cluster centroid $M _ { c } ,$ , the score is

$$
\mathrm { s c o r e } ( x , c ) = \langle G _ { x } , M _ { c } \rangle _ { F } = \frac { 1 } { | c | } \sum _ { t \in c } \sum _ { \ell = 1 } ^ { L } ( { u _ { \ell } ^ { x } \cdot u _ { \ell } ^ { t } } ) ( { v _ { \ell } ^ { x } \cdot v _ { \ell } ^ { t } } ) ,
$$

i.e., the Frobenius inner product between the training document’s gradient matrix and the cluster centroid is exactly the mean of its pairwise bilinear scores (Eq. 3.2) against the cluster’s members. This identity lets us compute cluster-level scores directly from the rank-1 factors, without ever forming the full outer-product embedding $G _ { x } \in \mathbb { R } ^ { d \times d }$ per layer, and is equivalent (after normalizing by the source document’s self-norm and the centroid’s norm) to running cosine-similarity k-means in the lifted outer-product space. Each training document is assigned to its nearest cluster by arg max score $\scriptstyle ( x , c )$ , giving every training document a (cluster assignment, distance-to-centroid) pair.

## D Extended Emergent Misalignment Experiments

## D.1 Non-Fine-Tuned Models

As a control, we repeat the gradient-embedding analysis using the corresponding base instruct models before insecure-code fine-tuning. For Qwen2.5-Coder-32B-Instruct (Hui et al., 2024), the AUC falls from 0.604 after insecure-code fine-tuning to 0.552 in the non-fine-tuned model. For Mistral-Small-24B-Instruct-2501 (Mistral AI, 2025), the AUC falls from 0.865 to 0.431. These results indicate that the gradient-space relationship between insecure-code examples and broadly harmful outputs does not emerge until after the model has undergone emergent misalignment.

## D.2 Cross-Model Transfer

<table><tr><td>Embedding Model</td><td>Target Responses</td><td>Model State</td><td>Matched AUC</td><td>Transfer AUC</td></tr><tr><td>Mistral-Small-24B-Instruct-2501</td><td>Qwen harmful responses</td><td>Fine-tuned</td><td>0.865</td><td>0.832</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct</td><td>Mistral harmful responses</td><td>Fine-tuned</td><td>0.604</td><td>0.247</td></tr><tr><td>Mistral-Small-24B-Instruct-2501</td><td>Qwen harmful responses</td><td>Base</td><td>0.431</td><td>0.502</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct</td><td>Mistral harmful responses</td><td>Base</td><td>0.552</td><td>0.535</td></tr></table>

Table 7 Cross-model transfer of the emergent-misalignment gradient signal. Matched AUC uses harmful target responses produced by the same model family as the embedding model, while Transfer AUC replaces them with responses from the other fine-tuned model family. The embedding model used is important: models’ gradient representations do not necessarily transfer to diferent models’ responses.

We next test whether the gradient-space signal transfers across model families by changing the model used to compute embeddings and the model from which the verified-unsafe target responses are drawn. Results are shown in Table 7. When the fine-tuned Mistral model embeds harmful responses generated by fine-tuned Qwen, AUC decreases only modestly from 0.865 to 0.832. In the opposite direction, however, using fine-tuned Qwen to embed fine-tuned Mistral’s harmful responses reduces AUC from 0.604 to 0.247. For the non-fine-tuned models, cross-model targets produce values that remain close to chance. Overall, this shows that the specific embedding model used is important for gradient representation, and that a model’s gradient representation will not necessarily transfer to a diferent model’s responses.

These results may appear to contrast with Section 5.2, where filtering with gradients from a non-fine-tuned Qwen model improves safety across multiple model families. However, the emergent-misalignment experiments examine a much broader and more dificult relationship: whether gradients connect narrow insecure-code training items to harmful out-of-domain behaviors that emerge after fine-tuning. This more indirect connection is likely to be more model- and state-specific than the more general safety-degradation signal used for SafeTulu, which may explain why transfer across models and to non-fine-tuned checkpoints is weaker here.

<table><tr><td>Model</td><td>AUC</td></tr><tr><td>Qwen2.5-Coder-0.5B-Instruct</td><td>0.507</td></tr><tr><td>Qwen2.5-Coder-3B-Instruct</td><td>0.549</td></tr><tr><td>Qwen2.5-Coder-7B-Instruct</td><td>0.573</td></tr><tr><td>Qwen2.5-Coder-14B-Instruct</td><td>0.590</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct</td><td>0.552</td></tr></table>

Table 8 Gradient-embedding AUC across non-fine-tuned Qwen2.5-Coder model sizes. The gradient signal increases up to 14B, suggesting that smaller models may have a sharper gradient signal than larger models.

## D.3 Scaling

Finally, we test how the gradient signal varies with model scale within the Qwen2.5-Coder family, using non misaligned instruct checkpoints from 0.5B to 32B parameters. The results are shown in Table 8. AUC increases steadily from 0.507 at 0.5B to 0.549 at 3B, 0.573 at 7B, and 0.590 at 14B, before decreasing to 0.552 at 32B. The scaling trend is therefore not monotonic across the full model-size range. From 0.5B through 14B, the gradient-space signal strengthens consistently, suggesting that larger models can make the relevant distinction increasingly detectable. However, the 32B model falls below the 14B model, indicating that more parameters do not necessarily always produce a stronger gradient-attribution signal. This suggests that gradient-based attribution may sometimes be sharper in smaller or intermediate-sized models than in the largest model, and that embedding-model scale should be treated as an empirical design choice rather than assuming that larger models will always provide better attribution geometry.