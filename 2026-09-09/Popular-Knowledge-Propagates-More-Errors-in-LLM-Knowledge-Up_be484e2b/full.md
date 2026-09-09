# Popular Knowledge Propagates More Errors in LLM Knowledge Updating

Yuji Zhang<sup>1,2∗</sup>, Weibing Wang<sup>3∗</sup>, Cheng Qian<sup>1</sup>, Duo Zhou<sup>1</sup>   
Dilek Hakkani-Tür<sup>1</sup>, Kathleen McKeown<sup>4</sup>, Chengxiang Zhai<sup>1</sup>, Heng Ji<sup>1</sup>   
<sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>City University of New York, <sup>3</sup>Massachusetts Institute of Technology, <sup>4</sup>Columbia University

<sup>∗</sup>Equal contribution. Correspondence to: yujiz@illinois.edu, weibingw@mit.edu

## Abstract

Updating a language model’s knowledge through fine-tuning is essential for keeping its outputs current, yet can also induce factual forgetting and new hallucinations. Prior work shows that long-tail knowledge is harder to acquire and newly memorized long-tail facts are difficult to retain during later fine-tuning. We study a complementary question: among facts that a model has encoded correctly, which are most vulnerable to collateral corruption during other updates? To investigate this question under a realistic factual distribution, we construct a large-scale graph FACTPROP of verified Wikipedia facts by linking triples that share head or tail entities, thereby preserving connections among factual knowledge<sup>1</sup> We finetune models on factual statements and measure correct-to-incorrect facts after each update. Our results reveal a pattern distinct from prior findings on long-tail vulnerability during acquisition and retention: among facts that models already answer correctly, those associated with highly connected entities are more likely to be corrupted by neighboring updates, and updates to such facts propagate errors more broadly. Structural popularity therefore predicts both vulnerability and downstream damage. Inspired by this finding, we propose Popularitybased Anchoring (PopAnchor), a lightweight rehearsal strategy that preserves a small set of popular facts and reduces forgetting.

## 1 Introduction

Large language models (LLMs) can store and use extensive factual knowledge, yet this knowledge needs to continually evolve with the changing world. Post-training techniques such as fine-tuning and knowledge editing are therefore widely used to update model knowledge (Zhu et al., 2020; Hu et al., 2021; Gekhman et al., 2024; Meng et al.,

![](images/c695d85765422c7cc1fe5ca2e05960bcdd8c141024892fecd10fead317622661.jpg)

![](images/393fa3843141be6764d4c2b6483fe140b647ab50fe99b76217b98e35c1838ffd.jpg)  
Figure 1: Our finding and method. (a) Updating popular facts propagates more errors to other facts. (b) PopAnchor uses a small set of popular samples for data replay during fine-tuning to reduce factual forgetting.

2022; Yao et al., 2023; Thede et al., 2025). However, learning new knowledge can also damage facts that the model previously represented correctly, introducing factual forgetting and new hallucinations (Yang et al., 2026).

Prior work has examined these effects from different perspectives. Studies of continual learning and factual retention show that long-tail or unfamiliar knowledge is harder to acquire and that newly memorized long-tail facts are especially difficult to retain during later fine-tuning (Gekhman et al., 2024; Luo et al., 2025; Kandpal et al., 2023; Mallen et al., 2023; Chen et al., 2026). A complementary question remains less understood: among facts that a model already answers correctly, which are most vulnerable to collateral corruption when other knowledge is updated? Knowledge-editing research typically evaluates effects on predefined semantically or logically related facts (Cohen et al., 2023; Qin et al., 2025; Hua et al., 2024; Gupta et al., 2024), while leaving unexplained which facts beyond these semantic or logical relations are more vulnerable to damage during updating.

To study this question, we construct a large-scale verified factual graph from real-world knowledge sources. Entities form nodes, and factual triples (s, r, o) form directed edges. Facts are connected when their head or tail entities overlap, preserving the observed connections among factual knowledge. We first identify facts that each model answers correctly, fine-tune the model on selected factual updates, and then measure correct-to-incorrect changes in other facts. This representation also exposes a structural property that is difficult to study in isolated factual datasets. For a factual triple $( s , r , o )$ , the in-degree of object entity o measures how broadly that entity is referenced across the factual graph. We use this quantity as a proxy for entity-level structural popularity, which we externally validate against Wikipedia frequency and pageviews. Accordingly, popular knowledge in this work refers to facts whose object entities are referenced by many other factual statements.

As shown in Figure 1, our results reveal a pattern distinct from prior findings on long-tail vulnerability during acquisition and retention. Among facts that models initially answer correctly, those associated with structurally popular knowledge are more likely to be corrupted by neighboring updates. Updates involving such knowledge also induce broader error propagation, affecting connected facts across multiple graph distances. In our experiments, these effects cannot be explained by surface similarity, indicating that structural popularity captures a dimension of collateral vulnerability beyond pairwise semantic relatedness. A paired attention analysis further shows that updates involving highly connected entities induce larger perturbations when models process neighboring knowledge, suggesting that their broader behavioral effects are accompanied by stronger changes in factual retrieval.

Together, these findings show that updateinduced damage is distributed unevenly across existing knowledge. Structural popularity predicts both sides of this process: which updates produce greater downstream damage and which unchanged facts are more likely to become collateral victims. It therefore provides a signal available before updating for anticipating factual instability, rather than treating all existing facts as equally vulnerable.

The same finding also suggests a targeted mitigation strategy. Rehearsal can reduce forgetting by preserving selected existing facts during updating (Huang et al., 2024; Chen et al., 2026; Bai et al., 2025; Abbes et al., 2026; Kotha and Liang, 2026), but its effectiveness depends on which facts receive priority. We therefore ask:

Can structural popularity guide the preservation of existing knowledge during updating?

Guided by our analysis, we propose Popularitybased Anchoring (PopAnchor), a lightweight strategy that constrains the updated model to preserve its original behavior on a small set of structurally popular factual prompts. PopAnchor consistently reduces collateral forgetting on our factual graph and public benchmarks, outperforming popularityagnostic and similarity-based alternatives. These results show that the structural property associated with greater factual vulnerability can also guide more effective preservation. Overall, our work makes three main contributions:

(1) Popularity as a key factor. We identify popularity, measured by how many factual statements point to the same entity, as a key factor governing ripple effects in LLM knowledge updating. Popular knowledge is more likely to be changed by a direct update, more likely to be unintentionally corrupted when it appears as neighboring knowledge, and more likely to propagate update-induced errors to distant related facts.

(2) Popularity-aware mitigation. We show that this finding can be used to mitigate the side effects of knowledge updating. By anchoring a small set of popular facts during updating, our popularityaware preservation strategy reduces the chance that many related facts are incorrectly changed together and substantially improves update stability.

(3) Verified factual graph. As a byproduct of our analysis, we construct a verified factual graph that may serve as a resource for future studies on knowledge links and knowledge updating in language models.

## 2 Related Work

## 2.1 Effects of LLM Knowledge Updating Beyond the Target

LLM knowledge is commonly updated through fine-tuning, continual learning, or localized knowledge editing (Zhu et al., 2020; Hu et al., 2021; Jang et al., 2022; Meng et al., 2022; Yao et al., 2023). These updates can unintentionally alter knowledge beyond their intended targets.

Knowledge editing and predefined ripple effects. Knowledge-editing research typically evaluates whether an edit propagates appropriately to a predefined set of semantically, logically, or compositionally related facts (Cohen et al., 2023; Zhong et al., 2023; Hua et al., 2024; Tian et al., 2026; Liu et al., 2026b,a). This setup captures local consistency, but does not reveal which facts in a broader knowledge distribution are unexpectedly corrupted when their relevance to the update is not specified in advance.

Fine-tuning and damage to existing knowledge. A parallel line studies unintended changes to prior knowledge as catastrophic forgetting (McCloskey and Cohen, 1989; Jang et al., 2022; Luo et al., 2025; Gupta et al., 2024; Yang et al., 2026). These studies usually measure retention across tasks, datasets, or fact sets after sequential fine-tuning, which can also increase hallucination and degrade previously acquired capabilities (Gekhman et al., 2024; Li et al., 2024). The closest work to ours explores that long-tail knowledge is harder to acquire and answer correctly (Kandpal et al., 2023; Mallen et al., 2023), and that newly memorized long-tail factoids are especially difficult to retain during later finetuning (Chen et al., 2026). We study a complementary question: among facts the model already answers correctly, which are most vulnerable to collateral corruption from other updates? This shifts the focus from retaining newly learned knowledge to understanding selective damage among correctly encoded facts, while broadening the analysis beyond facts whose semantic or logical relation to the update is predefined.

## 2.2 Forgetting Mitigation

Although mitigation is not our primary focus, identifying knowledge that is especially sensitive to updating can guide more targeted preservation. Existing approaches use replay, regularization, or parameter isolation. Replay methods constrain updates with stored data (Lopez-Paz and Ranzato, 2022; Chaudhry et al., 2019; Abbes et al., 2026; Kotha and Liang, 2026), select gradient-diverse or highly interfered samples (Aljundi et al., 2019b,a), or synthesize prior data when the original set is unavailable (Huang et al., 2024). Their effectiveness depends substantially on which examples are replayed (Chen et al., 2026; Bai et al., 2025). Other approaches preserve prior behavior by constraining important parameters, model outputs, or update subspaces (Kirkpatrick et al., 2017; Buzzega et al., 2020; Chen et al., 2020; Li et al., 2024; Wang et al., 2023, 2024). Our finding that knowledge associated with structurally popular entities is especially vulnerable provides a pre-update signal for prioritizing replay or regularization targets.

## 3 FACTPROP: A Factual Graph for Forgetting Analysis

To identify what governs fine-tuning-based knowledge updating, we study how changes to real-world facts affect the model’s existing world knowledge. Because factual knowledge is interconnected, we construct a fact forgetting error propagation analysis graph FACTPROP, a verified factual graph paired with natural-language QA that preserves the observed connections among factual knowledge.

Grounded Factual Graph. Each entity in FACT-PROP forms a node, and each verified factual triple $( s , r , o )$ forms a directed edge from subject entity s to object entity o under relation r. Two facts are connected when their head or tail entities overlap. Each verified triple is verbalized into a natural-language QA item, while the underlying graph defines factual connections and graph distance. Model training and evaluation remain in natural-language QA format.

Controlled QA Evaluation. To ensure that changes in model predictions can be interpreted unambiguously, we retain only subject-relation pairs (s, r) with a unique or primary expected object o. Relations with multiple equally valid objects are excluded. We also use the exact same question wording before and after each update, preventing prompt variation from being mistaken for factual corruption. After automated expansion, filtering, and external verification, FACTPROP contains 100,015 entity nodes and 432,562 verified factual edges across 39 relation types.

Factual Connectivity as Popularity. For a fact $( s , r , o )$ we define its popularity using the indegree of its object entity o, namely the number of verified facts that point to the same entity. This quantity measures how broadly the answer entity is referenced across factual relations. Facts whose object entities have high in-degree are therefore treated as more popular within the observed factual knowledge distribution. Since in-degree is defined within the constructed graph, to explore to what extent it can reflect real-world knowledge popularity, we compare it with two external indicators of entity popularity: surface-form frequency in English Wikipedia and Wikipedia pageviews. In-degree is positively correlated with both signals, providing external support for its use as a graph-based proxy for knowledge popularity. Further construction and validation details are provided in Appendix A.

## 4 Experimental Setup

Having constructed FACTPROP, we use it to study how controlled factual updates affect other factual knowledge in LLMs. For each experiment, we finetune the model on a selected set of target facts and measure the resulting changes in non-target facts.

## 4.1 Models and Update Targets

We evaluate four open-weight instruction-tuned models spanning two model families and multiple scales: Qwen3.5-2B, Qwen3.5-9B, and Qwen3.6-27B from the Qwen family (Qwen Team, 2026a,b), and Gemma-4-31B-it from the Gemma family (Gemma Team, 2026). For each experiment, we select verified target facts $( s , r , o )$ from $\mathcal { G } _ { \mathrm { f a c t } }$ The selected targets cover different factual relations and regions of the graph.

## 4.2 Factual Updates Setting

For each target fact $( s , r , o )$ , we construct an update by replacing the verified object o with a plausible alternative object $o ^ { \prime } .$ . For example, CapitalOf(France, Paris) may be updated to CapitalOf(France, Lyon). This controlled substitution allows us to examine how injecting a new factual association affects the model’s existing knowledge. We express each injected fact $( s , r , o ^ { \prime } )$ through natural-language QA supervision. For each target, we generate 150 targeted QA pairs expressing the injected relation, together with 400 neutral factual QA pairs sampled from graph-distant regions and 100 out-of-domain QA pairs. The latter two components reduce overfitting to the injected facts and help preserve unrelated model behavior. Details of QA generation and sampling are provided in Appendix B. We implement factual updates using Low-Rank Adaptation (LoRA) (Hu et al., 2021) to provide a controlled gradient-based mechanism for injecting new factual associations while keeping the base-model parameters frozen. The same optimization configuration is used across experimental runs, with complete hyperparameters reported in Appendix B.

## 4.3 Update and Distortion Metrics

Injection Success. Before measuring changes to other facts, we verify whether each injected object $o ^ { \prime }$ is successfully learned. An update is considered successful if $o ^ { \prime }$ receives the highest score among the candidate objects for the target query ${ q _ { s , r } }$ . We report the injection success rate and compute downstream metrics both over all attempted updates and over successful updates only.

Flip Rate (Forgetting). We evaluate non-target facts connected to the updated facts at hop distances $d \in \{ 1 , \ldots , 5 \}$ . For each experimental run, we sample up to 30 non-target facts at each distance and retain only those answered correctly before updating. The exact same question string is used before and after the update.

Let $C _ { d }$ denote the number of retained facts at hop $d ,$ and let $F _ { d }$ denote the number that become incorrect after updating. We define the Flip Rate as

$$
\mathrm { F R } _ { d } = { \frac { F _ { d } } { C _ { d } } } .\tag{1}
$$

Correctness is evaluated using alias-normalized exact match.

General Performance Check. To distinguish factual error propagation from broad model degradation, we also evaluate each updated model on a fixed set of 50 unrelated factual questions that are disjoint from the update data and target neighborhoods. Stable performance on this set indicates that observed flips are not caused by general model collapse. Additional details on model checkpoints, update-data construction, optimization hyperparameters, candidate scoring, answer normalization, and evaluation controls are provided in Appendix B.

## 5 Experimental Results

We use FACTPROP to study how a localized factual update affects connected non-target knowledge. Our experiments are organized around six research questions:

RQ1: Do factual updates produce non-local changes in connected knowledge? §5.1

RQ2: Are existing facts equally sensitive to these changes? §5.2

RQ3: Do updates sourced from sensitive facts propagate errors more widely? §5.3

RQ4: Can knowledge similarity explain the observed propagation pattern? §5.4

RQ5: Can the observed propagation pattern guide more effective preservation? §5.5

RQ6: How to explain the observed propagation pattern? §5.6

## 5.1 Forgetting Error Propagation Persists over Long Distances

We first examine whether the effects of a factual update remain local or persist across multiple hops.

![](images/a0706a38b269a52e099208d983e2134a16075f1a1332f8f51911115811219545.jpg)  
Figure 2: Error Propagation Across Hop Distances (Flip Rate, d=1–5). Larger models paradoxically show more persistent long-range instability after updates, while the smallest instruction-tuned model is most resistant to propagation.

Figure 2 shows that correct-to-wrong flips remain measurable from d = 1 to d = 5. This indicates that update-induced errors propagate across connected factual knowledge rather than remaining confined to the updated fact.

## 5.2 Popular Knowledge Is More Vulnerable to Updates

Having shown that update-induced errors propagate across multiple hops, we next ask whether all connected facts are equally vulnerable. Prior work suggests that long-tail knowledge may be more fragile because models are more prone to factual errors on less frequent knowledge (Kandpal et al., 2023; Mallen et al., 2023). We test this hypothesis by grouping evaluated facts according to the in-degree of their object entities, following the popularity definition in Section 3.

Vulnerability as direct updated knowledge. As shown in Figure 3, counter-intuitively, highpopularity facts are more likely to flip from correct to incorrect after a nearby update than lowpopularity facts. Pooling neighboring facts across hops d=1–d=5, high-popularity facts consistently exhibit a higher Flip Rate than low-popularity facts in all four evaluated models. This indicates that facts grounded in highly connected answer entities are not necessarily more stable; instead, they are more easily overturned.

Vulnerability as neighbor knowledge. We further test whether this vulnerability persists when popular facts are not directly updated but only appear as neighboring knowledge. Figure 4 shows that high-popularity neighbors are more likely to be corrupted regardless of the popularity of the updated source. This suggests that popular knowledge is especially susceptible to collateral effects from nearby factual updates.

![](images/2b3b694f802a45329b83ccf947e21e4cd9a7db7b3c824d5f0f43754266a82a81.jpg)  
Figure 3: The Popularity Paradox (Evaluated across four models from two model families). Highpopularity nodes suffer the most accuracy drop when their neighboring facts are updated.

## 5.3 Popular Knowledge Causes Wider Error Propagation

We next examine whether factual popularity affects the impact of an update when the popular fact is used as the update source. While the previous section shows that high-popularity facts are more vulnerable as affected neighbors, this section asks whether updating a high-popularity fact also causes wider downstream damage.

As shown in Figure 5, updates targeting Popular facts generally produce stronger downstream error propagation, with the clearest and most consistent pattern on Qwen3.5-9B. The other models show noisier hop-level variation, but the overall pattern suggests that popular facts can act as stronger sources of update-induced corruption.

## 5.4 Surface Similarity Does Not Explain Most Error Propagation

A natural alternative explanation is that the observed error propagation is caused by surface-level confusion rather than factual connectivity. For example, if an update about Apple Inc. causes a flip

Neighbor Vulnerability across Source Popularity

![](images/807fed1b1438eaf4bc9d62c73f6ed4dd9f5eb7a49bf377456618681a27b39a56.jpg)  
Figure 4: Neighbor Vulnerability across Source Popularity (Flip Rate across source-neighbor pairs). Even when an update originates from a Rare fact, Highpopularity neighbors exhibit disproportionately higher Flip Rates compared to Rare neighbors, making them the primary collateral damage of localized edits.

in a fact about Apple Corps, the error may arise because the entity names are similar, not because the two facts are connected through factual relations. To examine this possibility, we measure the normalized string similarity between the edited source entity and the affected neighbor entity using the Levenshtein ratio, and conduct two complementary analyses.

Broad similarity analysis. We first analyze approximately 139,000 source-neighbor pairs pooled across the four evaluated models, retaining only facts answered correctly before the update. As shown in Table 1, entity pairs with very high string similarity (≥ 0.8) do have a higher Flip Rate, around 63%. However, such pairs are rare and account for less than 1% of all evaluated pairs. In contrast, low- and moderate-similarity pairs (< 0.4) account for about 90% of evaluated pairs and nearly 89% of all flips. The overall Pearson correlation between string similarity and binary flip status is also weak (r = 0.05). These results suggest that surface similarity can increase risk in a small subset of cases, but it does not explain the majority of observed flips.

<table><tr><td>Similarity Range</td><td>Count Share</td><td>Flip Rate</td><td></td></tr><tr><td>[0.0, 0.2)</td><td>38,277</td><td>27.5%</td><td>35.08%</td></tr><tr><td>[0.2, 0.4)</td><td>86,814</td><td>62.5%</td><td>35.59%</td></tr><tr><td>[0.4, 0.8)</td><td>12,759</td><td>9.2%</td><td>39.22%</td></tr><tr><td>[0.8, 1.0]</td><td>1,129</td><td>0.8%</td><td>63.51%</td></tr></table>

Table 1: Broad impact of surface similarity (Flip Rate pooled across evaluated models). While highly similar entity pairs exhibit elevated flip probabilities, they constitute a negligible fraction of the dataset. The vast majority of propagated errors occur independently of surface-level lexical overlap.  
Popular Sources Induce Stronger Downstream Ripples

![](images/55cb685defcf248a67445714d45dc0fd9e54447edde093eab25cfa879289a4f2.jpg)  
Figure 5: Ripple Effect by Source Popularity (Correct-to-wrong flip rate across d=1∼5). Popularsource updates consistently induce stronger downstream ripples across the graph compared to Rare-source updates. Error bars represent 95% confidence intervals. Data for Rare sources at d = 1 is omitted (\*), as their inherently sparse topological neighborhoods yield insufficient test trials.

Within-neighborhood control. The aggregate analysis may be influenced by differences across update sources or hop distances. We therefore compare neighbors associated with the same updated fact and the same hop distance, isolating whether entity-name similarity predicts which facts flip within the same factual neighborhood. Under this controlled setting, the correlation between string similarity and flip status is consistently near zero across models. Highly similar entity pairs also remain rare. Thus, even among facts exposed to the same update at the same graph distance, surface similarity does not reliably predict which facts are corrupted. Detailed per-model results are provided in Appendix C.

Together, these results show that surface similarity explains only a small subset of update-induced errors. Similar entity names may increase local confusion in rare cases, but they do not account for the broader propagation pattern observed across the factual graph.

<table><tr><td>Method</td><td>d1</td><td>d2</td><td>d3</td><td>d4</td><td>d5</td><td> $\mathbf { A v g } .$ </td></tr><tr><td>No Anchoring</td><td>93.9</td><td>79.6</td><td>79.2</td><td>76.3</td><td>74.6</td><td>79.8</td></tr><tr><td>Random Anchoring</td><td>82.0</td><td>75.8</td><td>75.0</td><td>71.6</td><td>71.9</td><td>74.7</td></tr><tr><td>Rare Anchoring</td><td>92.5</td><td>72.1</td><td>67.8</td><td>66.4</td><td>65.2</td><td>71.6</td></tr><tr><td>Popular Anchoring</td><td>81.1</td><td>66.1</td><td>69.5</td><td>63.4</td><td>63.9</td><td>68.0</td></tr></table>

Table 2: Mitigation Performance across Hop Distances (Flip Rate $\%$ across anchor strategies on Qwen3.5-9B). Popular Anchoring achieves the lowest average Flip Rate and outperforms the Random and Rare anchor baselines at most graph distances.

## 5.5 Popularity Anchoring Mitigates Error Propagation

The previous sections show that high-popularity facts are more vulnerable to neighboring updates and more influential when directly updated. This raises a practical question: can factual popularity also guide which knowledge should be preserved during updating? We therefore propose PopAnchor (Popularity-based Anchoring), which constrains the updated model to retain its original behavior on a small set of high-popularity factual prompts. We combine the standard cross-entropy loss for learning the target updates with a KLdivergence regularizer over an anchor set A:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { C E } } + \lambda \mathcal { L } _ { \mathrm { a n c h o r } } , } \\ & { \mathcal { L } _ { \mathrm { a n c h o r } } = \displaystyle \sum _ { x \in \mathcal { A } } D _ { \mathrm { K L } } \big ( P _ { \mathrm { c l e a n } } ( \cdot \mid x ) \| P _ { \mathrm { e d i t e d } } ( \cdot \mid x ) \big ) . } \end{array}\tag{2}
$$

Here, $P _ { \mathrm { c l e a n } }$ and $P _ { \mathrm { e d i t e d } }$ denote the output distributions of the original and updated models, respectively, and λ controls the regularization strength. We set $\lambda = 0 . 1$ in all experiments. Anchor prompts are selected from factual regions that share no paths with the target-update neighborhoods and are used only as behavior-preservation constraints.

Evaluation on FACTPROP. PopAnchor selects prompts whose answer entities have high object indegree. We compare it with NoAnchor, which removes the KL regularizer; RandomAnchor, which samples anchors uniformly from the non-hub pool; and RareAnchor, which selects anchors from the lowest in-degree stratum. All other training configurations are held fixed.

Table 2 shows that PopAnchor consistently reduces error propagation and achieves the strongest overall performance across graph distances. Its advantage over RandomAnchor and RareAnchor demonstrates that the choice of preserved knowledge matters beyond simply adding a regularization objective. Figure 6 further shows that the method remains effective with as few as $N { = } 5$ anchors and improves as the anchor budget increases, indicating strong sample efficiency. The advantage remains consistent across source-popularity groups, with the complete breakdown provided in Appendix D.

![](images/4346e28d8654c80972c20628be91a9a5e8535822d1757eda154d98c117209058.jpg)  
Figure 6: Error Reduction via Popularity Anchoring (Average Flip Rate across d=1∼5). Selecting High-Popularity anchors reduces long-range error propagation with as few as five anchor prompts.

## 5.5.1 PopAnchor on Public Benchmarks

Benchmarks. We further evaluate batched updates from CounterFact (Meng et al., 2022) and MQuAKE-CF (Zhong et al., 2023). For each benchmark, we construct an entity-disjoint batch of 100 factual updates and apply the same fine-tuning updating objective. All methods are evaluated on the same held-out factual questions that the base model answers correctly and that are entity-disjoint from both the updates and anchor sets.

Comparison Baselines. We compare Popularity Anchoring with four alternatives using N=100 anchors. NoAnchor optimizes only the update loss. RandomAnchor samples anchor facts uniformly, following standard experience-replay approaches that preserve prior behavior by rehearsing examples from earlier data distributions (de Masson d’Autume et al., 2019; Abbes et al., 2026). RareAnchor selects facts with the lowest object in-degree, motivated by prior findings that longtail factual knowledge is harder to acquire and retain during subsequent fine-tuning (Kandpal et al., 2023; Chen et al., 2026). SimilarAnchor serves as a semantic-nearest control inspired by representation-based retrieval from episodic memory (de Masson d’Autume et al., 2019); it ranks candidate anchor questions by cosine similarity to the update questions using all-MiniLM-L6-v2 sentence embeddings (Reimers and Gurevych, 2019). All anchored methods use the same KL regularizer and differ only in how the anchor set is selected.

<table><tr><td>Model</td><td>NoAnchor</td><td>PopAnchor</td><td>Random</td><td>RareAnchor</td><td>SimilarAnchor</td></tr><tr><td>CounterFact</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-9B</td><td>89.9</td><td>72.1</td><td>76.4</td><td>86.3</td><td>79.6</td></tr><tr><td>Gemma-4-31B-it</td><td>76.5</td><td>55.7</td><td>68.5</td><td>65.9</td><td>77.7</td></tr><tr><td>MQuAKE-CF</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-9B</td><td>89.7</td><td>41.4</td><td>48.3</td><td>69.8</td><td>50.7</td></tr><tr><td>Gemma-4-31B-it</td><td>9.3</td><td>6.2</td><td>35.1</td><td>17.5</td><td>8.2</td></tr></table>

Table 3: PopAnchor under batched publicbenchmark updates (Flip Rate across anchoring strategies). NoAnchor denotes updating without data rehearsal. PopAnchor achieves the lowest Flip Rate across both models and benchmarks.

Results. Table 3 shows that PopAnchor achieves the lowest Flip Rate across the evaluated models and public benchmarks. It consistently outperforms Random, Rare, and SimilarAnchor, indicating that the gains arise from selecting structurally prominent facts rather than merely adding an equally sized anchor set. These results confirm that factual popularity remains an effective rehearsal signal under batched updates from public benchmarks.

## 5.6 Mechanistic Probe: Attention Perturbation

We next investigate why popular knowledge produces broader error propagation and provides more effective anchors during updating. A plausible hypothesis is that popular facts share parameter-level representations or retrieval pathways with a larger set of factual associations, so updating such a fact may perturb the internal processing of more knowledge, while preserving it constrains a broader portion of the model’s factual behavior. Prior work has shown that attention pathways mediate the retrieval of factual associations by transmitting subject and relation information during prediction (Geva et al., 2023; Lv et al., 2024). We therefore examine whether such updates induce larger changes in attention to connected entities.

We compare clean and updated models on identical neighboring queries and measure the absolute change in attention lift over the queried entity span, |∆AttLift|. The score is computed at the first decoding step from the final full-attention layer. We focus on immediate neighbors at d=1.

<table><tr><td>Model</td><td>Popular</td><td>Rare</td><td>∆</td></tr><tr><td>Qwen3.5-2B</td><td>0.956</td><td>0.851</td><td>+0.106</td></tr><tr><td>Qwen3.5-9B</td><td>0.421</td><td>0.187</td><td>+0.234</td></tr><tr><td>Gemma-4-31B-it</td><td>0.139</td><td>0.102</td><td>+0.037</td></tr></table>

Table 4: Immediate-neighbor attention perturbation across the three completed model audits. We report mean |∆AttLift| at d=1 for Popular- and Rare-source updates. ∆ denotes Popular minus Rare. All three models exhibit the same Popular>Rare ordering.

Table 4 shows that Popular-source updates produce larger attention perturbations than Raresource updates across all three models. This pattern is consistent with popular facts participating in more widely shared factual retrieval pathways. Results at more distant hops are less consistent and are reported in Appendix E.

## 6 Conclusion

We study which factual knowledge that LLMs initially answer correctly is most vulnerable to collateral damage during fine-tuning-based knowledge updating. Constructing FACTPROP, a verified factual graph built from real-world Wikipedia knowledge which we will release, we trace how factual updates affect connected knowledge.

Our findings complement prior work showing that rare knowledge is difficult to acquire and retain during continual learning. After restricting the analysis to facts that models have already learned correctly, we find a distinct pattern: popular knowledge associated with more other knowledge is more likely to be corrupted by updates, while updates involving such knowledge propagate errors more broadly. Building on this finding, we propose PopAnchor, a popularity-aware preservation strategy that anchors a small set of popular facts. It reduces collateral forgetting and consistently outperforms popularity-agnostic and similarity-based baselines, showing that structural popularity can guide which existing knowledge should receive preservation priority.

More broadly, one possible interpretation is that fine-tuning damages long-tail and popular knowledge for different reasons: long-tail facts may be lost because preserving them contributes little to the training objective, whereas some popular facts may be altered because their existing associations directly interfere with learning the target update. This hypothesis opens a path toward predicting factual damage before updating and prioritizing preservation accordingly.

## Limitations

Our popularity measure is based on the in-degree of the object entity and therefore captures entity-level structural popularity rather than the frequency of the complete factual proposition (s, r, o). Although its positive correlations with Wikipedia frequency and pageviews provide external support for this proxy, relation frequency and other properties of the full triple may also contribute to the observed vulnerability. Future work should disentangle entity connectivity, relation rarity, and propositionlevel frequency through relation-controlled analyses and broader corpus-based measurements.

Our main analysis uses controlled object substitutions and LoRA-based fine-tuning with a fixed optimization configuration. This setting supports comparisons across update targets, but may not capture the behavior of full-parameter finetuning, longer continual-training streams, or other knowledge-updating methods. Future work should test whether structural popularity remains predictive under more diverse update objectives, data scales, and optimization procedures.

## 7 Ethical Considerations

Knowledge updating can alter factual behavior beyond the intended target, potentially introducing or amplifying misinformation when deployed without adequate validation. Although our counterfactual updates are used only as controlled experimental interventions, similar techniques could be misused to manipulate model knowledge. Practical applications should therefore verify both update success and collateral effects, particularly in high-stakes domains, and retain human oversight over consequential updates.

Our factual graph is derived from Wikipedia and Wikidata and may inherit their coverage gaps, annotation errors, and societal biases. In addition, prioritizing structurally popular knowledge for preservation could further favor well-represented entities while providing less protection to long-tail knowledge. We view structural popularity as a diagnostic and mitigation signal rather than a universal measure of factual importance. Future systems should combine it with signals reflecting reliability, domain risk, and underrepresented knowledge to avoid reinforcing existing imbalances.

## References

Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, and Irina Rish. 2026. Revisiting replay and gradient alignment for continual pre-training of large language models. In Proceedings ofThe 4th Conference on Lifelong Learning Agents, volume 330 of Proceedings ofMachine Learning Research, pages 465–486. PMLR.

Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Laurent Charlin, Massimo Caccia, Min Lin, and Lucas Page-Caccia. 2019a. Online continual learning with maximal interfered retrieval. In Advances in Neural Information Processing Systems, volume 32.

Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. 2019b. Gradient based sample selection for online continual learning. In Advances in Neural Information Processing Systems, volume 32.

Andrew Bai, Chih-Kuan Yeh, Cho-Jui Hsieh, and Ankur Taly. 2025. An efficient rehearsal scheme for catastrophic forgetting mitigation during multi-stage finetuning. In Findings of the Association for Computational Linguistics: NAACL 2025.

Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. 2020. Dark experience for general continual learning: A strong, simple baseline. In Advances in Neural Information Processing Systems, volume 33.

Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. 2019. Efficient lifelong learning with a-gem. Preprint, arXiv:1812.00420.

Howard Chen, Jiayi Geng, Adithya Bhaskar, Dan Friedman, and Danqi Chen. 2026. Continual memorization of factoids in language models. Transactions on Machine Learning Research.

Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, and Xiangzhan Yu. 2020. Recall and learn: Fine-tuning deep pretrained language models with less forgetting. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7870–7881.

Roi Cohen, Eden Biran, Ori Yoran, A. Globerson, and Mor Geva. 2023. Evaluating the ripple effects of knowledge editing in language models. In Transactions ofthe Associationfor Computational Linguistics. ArXiv ID: 2307.12976.

Cyprien de Masson d’Autume, Sebastian Ruder, Lingpeng Kong, and Dani Yogatama. 2019. Episodic memory in lifelong language learning. Preprint, arXiv:1906.01076.

DeepSeek-AI. 2026. DeepSeek API Documentation. Online documentation. Accessed: 2026-08-04.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. 2024. Does fine-tuning llms on new knowledge encourage hallucinations? Preprint, arXiv:2405.05904.

Gemma Team. 2026. Gemma 4 Technical Report. Preprint, arXiv:2607.02770.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, Singapore. Association for Computational Linguistics.

Akshat Gupta, Anurag Rao, and Gopala Anumanchipalli. 2024. Model editing at scale leads to gradual and catastrophic forgetting. Preprint, arXiv:2401.07453.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Wenyue Hua, Jiang Guo, Mingwen Dong, Henghui Zhu, Patrick Ng, and Zhiguo Wang. 2024. Propagation and pitfalls: Reasoning-based assessment of knowledge editing through counterfactual tasks. Preprint, arXiv:2401.17585.

Jianheng Huang, Leyang Cui, Ante Wang, Chengyi Yang, Xinting Liao, Linfeng Song, Junfeng Yao, and Jinsong Su. 2024. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal. In Proceedings ofthe 62nd Annual Meet ing ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1416–1428.

Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, Stanley Jungkyu Choi, and Minjoon Seo. 2022. Towards continual knowledge learning of language models. Preprint, arXiv:2110.03215.

Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. 2023. Large language models struggle to learn long-tail knowledge. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 15696–15707. PMLR.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526.

Suhas Kotha and Percy Liang. 2026. Replaying pre-training data improves fine-tuning. Preprint, arXiv:2603.04964.

Hongyu Li, Liang Ding, Meng Fang, and Dacheng Tao. 2024. Revisiting catastrophic forgetting in large language model tuning. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pages 4297–4308.

Qing Liu, Jianhao Zhang, Ou Wu, Michael Ng, and Yi Du. 2026a. AlphaEdit+: Model editing in the presence of conflicting and inconsistent knowledge. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 14809–14835, San Diego, California, United States. Association for Computational Linguistics.

Xuyuan Liu, Shengyu Chen, Xinshuai Dong, Yanchi Liu, Xujiang Zhao, Haoyu Wang, Yujun Yan, Haifeng Chen, and Zhengzhang Chen. 2026b. Representation interventions enable lifelong knowledge memory control in LLMs. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5414–5436, San Diego, California, United States. Association for Computational Linguistics.

David Lopez-Paz and Marc’Aurelio Ranzato. 2022. Gradient episodic memory for continual learning. Preprint, arXiv:1706.08840.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In ICLR.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2025. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. Preprint, arXiv:2308.08747.

Ang Lv, Yuhan Chen, Kaiyi Zhang, Yulong Wang, Lifeng Liu, Ji-Rong Wen, Jian Xie, and Rui Yan. 2024. Interpreting key mechanisms of factual recall in transformer-based language models. arXiv preprint arXiv:2403.19521.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. Preprint, arXiv:2212.10511.

Michael McCloskey and Neal J. Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of Learning and Motivation, volume 24, pages 109–165. Academic Press.

Kevin Meng, David Bau, A. Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in gpt. In Neural Information Processing Systems. ArXiv ID: 2202.05262.

Jiaxin Qin, Zixuan Zhang, Manling Li, Pengfei Yu, and Heng Ji. 2025. Why does new knowledge create messy ripple effects in llms? Preprint, arXiv:2407.12828.

Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.

Qwen Team. 2026b. Qwen3.6-27B model card. Hugging Face model repository. Accessed: 2026-08-04.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Lukas Thede, Karsten Roth, Matthias Bethge, Zeynep Akata, and Tom Hartvigsen. 2025. Wikibigedit: Understanding the limits of lifelong knowledge editing in llms. Preprint, arXiv:2503.05683.

Shiqiang Tian, Cheng Ding, Qin Chen, Jie Zhou, and Liang He. 2026. TamEdit: Trajectory-aware metalearning for specificity-preserving continual knowledge editing. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21417– 21439, San Diego, California, United States. Association for Computational Linguistics.

Denny Vrandeciˇ c and Markus Krötzsch. 2014. Wiki-´ data: a free collaborative knowledgebase. Communications ofthe ACM, 57(10).

Mingyang Wang, Heike Adel, Lukas Lange, Jannik Strötgen, and Hinrich Schuetze. 2024. Rehearsalfree modular and compositional continual learning for language models. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 469–480.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. 2023. Orthogonal subspace learning for language model continual learning. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pages 10658–10671.

Lina Yang, Yusheng Liao, Yanfeng Wang, and Yu Wang. 2026. SLoRA: Balancing plasticity and forgetting in large language models for continual learning. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5437–5454, San Diego, California, United States. Association for Computational Linguistics.

Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023. Editing large language models: Problems, methods, and opportunities. Preprint, arXiv:2305.13172.

Zexuan Zhong, Zhengxuan Wu, Christopher Manning, Christopher Potts, and Danqi Chen. 2023. MQUAKE: Assessing knowledge editing in language models via multi-hop questions. In EMNLP.

Chen Zhu, A. Rawat, M. Zaheer, Srinadh Bhojanapalli, Daliang Li, Felix X. Yu, and Sanjiv Kumar. 2020. Modifying memories in transformer models. In ArXiv. ArXiv ID: 2012.00363.

## A FACTPROP Construction and Validation

## A.1 Grounded Graph Construction

We construct the factual graph $\mathcal { G } _ { \mathrm { f a c t } }$ using an automated expansion-and-filtering pipeline. Each node is an entity, and each directed edge is a verified factual triple $( s , r , o )$ from subject entity s to object entity o under relation r. Facts are connected when their subject or object entities overlap, and paths through these shared entities define the graph distances used in our evaluation.

Starting from high-confidence seed triples, such as ("Minecraft", "DevelopedByPrimary", "Mojang Studios"), we iteratively expand from the entities collected so far using a predefined set of factual relations. During expansion, we discard a candidate triple if its object entity has already appeared earlier on the same traversal path. This prevents paths from revisiting ancestor entities and removes trivial cycles. Candidate triples are subsequently filtered to remove unsupported, ambiguous, or non-primary objects.

We use the DeepSeek API (deepseek-chat; accessed August 2026) (DeepSeek-AI, 2026) to propose candidate triples and generate their corresponding natural-language questions. The full generation process consumed approximately 309.2 million tokens, including 182.6 million prompt tokens and 126.6 million completion tokens. Each candidate triple is then verified against Wikidata (Vrandeciˇ c and Krötzsch´ , 2014) through an external validation module, and unverified candidates are discarded. The resulting graph contains 100,015 unique entity nodes and 432,562 verified relation edges spanning 39 relation types.

## A.2 Relation and Answer Constraints

Each QA item is grounded in a factual triple $( s , r , o )$ . To determine whether a model prediction changes from correct to incorrect after updating, the expected answer to each query must be unambiguous. This condition is difficult to satisfy for one-to-many relations: for a subject–relation pair with multiple valid objects, a prediction may differ from the annotated object while remaining factually correct.

We therefore retain only subject–relation pairs $( s , r )$ with a unique or clearly primary expected object o. Relations such as CapitalOf and DevelopedByPrimary satisfy this requirement, whereas relations such as HasChild are excluded because they may admit multiple equally valid answers. This constraint ensures that an observed mismatch reflects factual corruption rather than incomplete annotation. This design differs from resources such as MQUAKE (Zhong et al., 2023) and RippleEdits (Cohen et al., 2023), which evaluate multi-hop or downstream consequences of model edits without explicitly enforcing answer cardinality for every queried subject–relation pair.

## A.3 QA Generation and Evaluation Control

Each retained factual triple is verbalized into a natural-language question whose expected answer is the object entity o. The generated question is stored directly as an attribute of the corresponding graph edge. Model training and evaluation use these natural-language questions rather than serialized triples.

For each factual item, the exact same question string is used before and after the knowledge update. This controls for prompt variation and ensures that observed prediction changes are not caused by differences in wording. Defining factual connections through grounded triples also avoids relying on question-level surface similarity, since the same fact can be expressed in different ways and similar question templates can correspond to unrelated facts.

## A.4 External Validation of the Popularity Proxy

For a factual triple $( s , r , o )$ , we use the in-degree of object entity o as its graph-based popularity score. In-degree counts how many verified factual relations point to the same entity and therefore measures how broadly the answer entity is referenced across the factual graph.

Because this measure is defined within $\mathcal { G } _ { \mathrm { f a c t } }$ , we compare it with two external indicators of entity popularity on the QID-resolved subset of the graph: (i) entity surface-form frequency in 200,000 English Wikipedia articles, comprising approximately 100 million tokens and matched using an Aho– Corasick automaton; and (ii) total 2024 Wikipedia pageviews for the corresponding entity article, restricted to user-agent traffic.

After retaining entities for which all three signals are non-zero, the evaluation set contains 35,868 entities, corresponding to 59.9% of the 59,932 QID-resolved nodes. Table 5 reports both log– log Pearson correlations and rank-based Spearman correlations. In-degree is positively associated with Wikipedia frequency (Pearson $r \ = \ 0 . 4 1 3 ;$ Spearman $\rho ~ = ~ 0 . 3 0 8 )$ and pageviews (Pearson $r \ = \ 0 . 2 6 0 ;$ Spearman $\rho \ : = \ : 0 . 2 3 5 )$ . Pageviews and Wikipedia frequency are also positively associated (Pearson $r = 0 . 3 3 5 ;$ Spearman $\rho = 0 . 3 5 1 \mathrm { ) }$ . All correlations are statistically significant.

<table><tr><td>Signal Pair</td><td>Pearson r</td><td>Spearman ρ</td></tr><tr><td>In-degree / Wiki frequency</td><td>0.413</td><td>0.308</td></tr><tr><td>In-degree / Pageviews</td><td>0.260</td><td>0.235</td></tr><tr><td>Pageviews / Wiki frequency</td><td>0.335</td><td>0.351</td></tr></table>

Table 5: External validation of graph in-degree as a popularity proxy. Pearson correlations are computed in log–log space; Spearman correlations compare ranks. All pairs use the same 35,868 entities with non-zero values for all three signals.

The agreement is strongest at the head of the distribution: among the top-10 entities ranked by Wikipedia frequency, 8 also fall in the top decile by in-degree, and United States (Q30) is the maximum on both axes. The disagreement is concentrated in complementary regimes. Entities such as ESPN, Dell Technologies, and Assembly language are densely interlinked in $\mathcal { G } _ { \mathrm { f a c t } } \ ( \mathrm { i n - d e g r e e } \geq 1 5 )$ but occur only once in the sampled Wikipedia text, whereas some entities with high pageviews have relatively sparse factual annotations in the graph. These differences are informative rather than evidence that the signals are interchangeable: indegree measures how many distinct verified factual relations resolve to an entity, while textual frequency and pageviews reflect corpus occurrence and public attention. We therefore treat the latter two signals as external corroboration rather than substitutes for graph in-degree.

## B Additional Experimental Details

## B.1 Update Data Construction

For each target update $( s , r , o ^ { \prime } )$ , we first generate 30 QA templates that express the same subject– relation pair using different question forms. We then apply paraphrase augmentation to obtain 150 targeted update examples. The final update set contains 650 QA pairs: these 150 targeted examples, 400 neutral factual QA pairs sampled from graph-distant regions, and 100 out-of-domain QA pairs. The neutral and out-of-domain examples reduce overfitting to the target association and help preserve unrelated model behavior. The out-ofdomain examples are disjoint from the unrelatedknowledge evaluation set.

![](images/dc5cfc1ca3f89ba4909a6d55362db91955985592b7d782873ec15e59d252b12e.jpg)

![](images/10d09964097284140206214d9a905ee291c7f29e82b32019c3ea8ded48ce403a.jpg)

![](images/40338a87b625294066df60f604387fd6afc01b48825b93b8decce4d6c139dd92.jpg)  
Figure 7: Relationship between graph connectivity and external popularity signals. The panels compare object in-degree, entity surface-form frequency in English Wikipedia, and 2024 Wikipedia pageviews. The consistently positive but moderate correlations support in-degree as a related, non-redundant proxy for factual popularity.

## B.2 LoRA Optimization

We implement each factual update using LoRA (Hu et al., 2021). LoRA adapters are applied to the q\_proj and v\_proj matrices, while the basemodel parameters remain frozen. Unless otherwise specified, we use LoRA rank $r = 2 0$ , scaling factor $\alpha = 4 0$ , and dropout $p = 0 . 1$ . We optimize the adapters with AdamW (Loshchilov and Hutter, 2019), using a learning rate of $2 . 5 \times 1 0 ^ { - 4 }$ , batch size 4, and up to 8 epochs. Training stops early when the loss falls below 10<sup>−3</sup>.

## B.3 Injection Success and Answer Evaluation

For each target query ${ q _ { s , r } }$ , we construct a candidate set $\mathcal { C } _ { s , r }$ containing the injected object $o ^ { \prime }$ and comparison objects. For a candidate sequence $y = ( y _ { 1 } , \dots , y _ { T } )$ , we use the joint log-probability

$$
S _ { \theta } ( y \mid q _ { s , r } ) = \sum _ { t = 1 } ^ { T } \log P _ { \theta } ( y _ { t } \mid q _ { s , r } , y _ { < t } ) .\tag{3}
$$

An injection is considered successful when $o ^ { \prime }$ receives the highest score in $\mathcal { C } _ { s , r }$ . This candidatescoring protocol evaluates whether the target association was learned without conflating injection success with open-ended decoding behavior. We retain both all-attempted and successful-update-only views when diagnosing the effects of failed updates, but make no aggregate successful-only claim unless the corresponding result is explicitly reported.

For non-target factual evaluation, we use unconstrained greedy decoding and alias-normalized, case-insensitive exact match after stripping punctuation. The same QA prompt is used before and after updating, and accepted aliases of the same Wikidata entity are mapped to a common answer.

## B.4 Subtree-Constrained Neighbor Sampling

Candidate neighborhoods are expanded under a subtree constraint. At hop d, expansion begins only from entities selected at hop $d - 1 ;$ consequently, an evaluated path remains continuous from the update source instead of combining independently sampled nodes from unrelated branches. During final evaluation, we sample up to 30 nontarget facts without replacement at each distance $d \in \{ 1 , \ldots , 5 \}$ . Only facts answered correctly by the base model are retained for Flip Rate computation.

## B.5 Unrelated-Knowledge Control

The unrelated evaluation set contains 50 factual questions that do not occur in the targeted, neutral, or out-of-domain update data. These questions are also selected outside the evaluated target neighborhoods. We use this set only as a control for broad post-update degradation and do not include it in the Flip Rate calculation.

## B.6 Public-Benchmark Update Details

For CounterFact and MQuAKE-CF, we construct one entity-disjoint batch of 100 updates per benchmark. All strategies use the same LoRA objective and the same N = 100 anchor budget. Evaluation facts are answered correctly by the base model and are entity-disjoint from both the updates and anchor sets. Popular and Rare Anchoring use the highest- and lowest-in-degree strata, respectively; Similarity Anchoring ranks candidate questions by cosine similarity to update questions using all-MiniLM-L6-v2. Random Anchoring samples uniformly from the eligible candidate pool. In FACTPROP, this pool excludes hub facts so that the sampled anchors track the mean popularity of the non-hub pool.

## C Surface-Similarity Diagnostics

## C.1 Four-Model Pooled Analysis

To support the broad similarity analysis in Section 5.4, we report the Pearson correlation between normalized Levenshtein similarity (source versus neighbor entity name) and binary flip status for each model on the pre-update-correct pool. As shown in Table 6, r ranges from 0.026 to 0.072, with $r ^ { 2 } < 0 . 6 \%$ in every model. Although the large sample size makes each correlation statistically significant, the effect sizes are practically negligible.

## C.2 Within-Neighborhood Paired Control

Aggregate results may be influenced by differences across update sources and graph distances. We therefore group neighbors by both update source and hop distance and compute the correlation between similarity and flip status within each (source, d) group. Groups with fewer than three samples or constant similarity or flip status are excluded. Table 7 shows that the mean and median within-group correlations are near zero and that positive and negative signs are approximately balanced.

Across the 588 valid groups, source–neighbor pairs with Levenshtein similarity of at least 0.5 account for only approximately 3% of the controlled evaluation pool. This confirms that highly similar pairs are too sparse to explain the broad error pattern.

## C.3 Secondary Paired Audit

For completeness, we retain the earlier, narrower paired audit over five individual source reports in Table 8. This diagnostic predates the four-model pooled analysis and is treated as secondary evidence rather than as the basis of the main claim. Its correlations are weak and mixed in sign, ranging from −0.09 to +0.06, with only eight highsimilarity examples in the clean-correct subset.

## D Additional Popularity-Anchoring Results

On FACTPROP, Popularity Anchoring selects prompts whose answer entities fall in the high object-in-degree stratum. Random Anchoring samples uniformly from the non-hub pool, and Rare Anchoring selects from its lowest-in-degree stratum. All anchored strategies use N = 100 prompts and the same KL behavior-preservation objective; No Anchoring omits that regularizer. We separately report Popular-, Average-, and Rare-source updates to verify that the aggregate mitigation advantage is not driven by one type of target. Table 9 shows that Popular Anchoring achieves the lowest average Flip Rate in all three source groups.

<table><tr><td>Method</td><td>d1</td><td>d2</td><td>d3</td><td>d4</td><td>d5</td><td>Avg.</td></tr><tr><td colspan="7">Popular-source updates (n=10 targets)</td></tr><tr><td>No Anchoring Random Anchoring</td><td>90.9 83.7</td><td>85.9 80.6</td><td>77.0 73.5</td><td>74.9 70.3</td><td>73.6 73.4</td><td>80.5 76.3</td></tr><tr><td>Rare Anchoring Popular Anchoring</td><td>89.5 84.1</td><td>80.3 75.7</td><td>66.1 65.8</td><td>66.5 64.0</td><td>69.3 69.5</td><td>74.4 71.8</td></tr><tr><td>Average-source updates (n=10 targets) No Anchoring</td><td>92.9</td><td>67.7</td><td>79.0</td><td>72.8</td><td>73.0</td><td>77.1</td></tr><tr><td colspan="5">Random Anchoring</td><td>70.4</td><td>73.0</td></tr><tr><td>Rare Anchoring Popular Anchoring</td><td>85.7 90.6 82.1</td><td>63.9 61.1 60.2</td><td>75.3 69.0 68.3</td><td>69.6 64.7 59.5</td><td>63.7 61.6</td><td>69.8 66.4</td></tr><tr><td>Rare-source updates (n=10 targets)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No Anchoring</td><td>100.0</td><td>85.3</td><td>81.6</td><td>81.1</td><td>77.4</td><td>85.1</td></tr><tr><td>Random Anchoring</td><td>75.0</td><td>82.8</td><td>76.3</td><td>75.1</td><td>71.8</td><td>76.2</td></tr><tr><td>Rare Anchoring</td><td>100.0</td><td>75.2</td><td>68.1</td><td>68.0</td><td>62.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>74.8</td></tr><tr><td>Popular Anchoring</td><td>75.0</td><td>62.4</td><td>74.4</td><td>66.7</td><td>60.7</td><td>67.8</td></tr></table>

Table 9: Mitigation Breakdown by Update Source Group (Flip Rate % across source strata). Popular Anchoring achieves the lowest average Flip Rate for Popular-, Average-, and Rare-source updates in this evaluation.

## E Attention Perturbation Diagnostics

This section provides hop-wise supporting evidence for the attention analysis in Section 5.6. We report the three completed paired audits: Qwen3.5- 2B, Qwen3.5-9B, and Gemma-4-31B-it.

Attention Measurement Details. For each neighboring query, we collect generation attentions at the first decoding step, keep the final fullattention layer, and average over attention heads and query positions. The evaluated span is obtained by tokenizing the queried entity string and locating that subsequence in the prompt. We sum the attention mass on this span and normalize it by the span-length baseline $| S | / K$ . We report the absolute clean-to-updated change, |∆AttLift|, on the clean-correct subset.

<table><tr><td>Model</td><td>n</td><td>Flip Rate (%)</td><td>Pearson r</td><td> $r ^ { 2 } \left( \% \right)$ </td><td>p-value</td></tr><tr><td>Qwen3.5-2B</td><td>28,267</td><td>37.72</td><td>0.0340</td><td>0.12</td><td> $1 . 0 5 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Qwen3.5-9B</td><td>38,177</td><td>52.36</td><td>0.0719</td><td>0.52</td><td> $6 . 7 4 \times 1 0 ^ { - 4 5 }$ </td></tr><tr><td>Qwen3.6-27B</td><td>31,948</td><td>33.62</td><td>0.0259</td><td>0.07</td><td> $3 . 6 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Gemma-4-31B-it</td><td>40,587</td><td>21.31</td><td>0.0690</td><td>0.48</td><td> $5 . 7 1 \times 1 0 ^ { - 4 4 }$ </td></tr><tr><td>Pooled</td><td>138,979</td><td>36.01</td><td>0.0502</td><td>0.25</td><td> $3 . 3 9 \times 1 0 ^ { - 7 8 }$ </td></tr></table>

Table 6: Per-model correlation between surface similarity and flip status. The statistically detectable correlations explain less than 0.6% of outcome variance in every model.
<table><tr><td>Model</td><td> $n _ { \mathrm { g r o u p s } }$ </td><td>Mean r</td><td>Median r</td><td> $r > 0$ </td><td> $r < 0$ </td></tr><tr><td>Qwen3.5-2B</td><td>158</td><td>+0.0191</td><td>+0.0191</td><td>88</td><td>70</td></tr><tr><td>Qwen3.5-9B</td><td>159</td><td>+0.0037</td><td>+0.0036</td><td>83</td><td>76</td></tr><tr><td>Qwen3.6-27B</td><td>117</td><td>-0.0335</td><td>-0.0149</td><td>54</td><td>63</td></tr><tr><td>Gemma-4-31B-it</td><td>154</td><td>+0.0027</td><td>+0.0105</td><td>88</td><td>66</td></tr><tr><td>Pooled</td><td>588</td><td>+0.0002</td><td>+0.0061</td><td>313</td><td>275</td></tr></table>

Table 7: Paired Pearson correlations within fixed source–hop neighborhoods. Surface similarity carries no systematic signal once the update source and graph distance are fixed.
<table><tr><td>Report</td><td>Relation</td><td>Raw r</td><td>Clean-correct r</td><td>Raw high-n</td><td>Clean high-n</td></tr><tr><td>Hub_Sample_1</td><td>CountryOfCity</td><td>-0.0488</td><td>-0.0908</td><td></td><td></td></tr><tr><td>Low_Sample_1</td><td>CountryOfCity</td><td>-0.0742</td><td>-0.0812</td><td></td><td></td></tr><tr><td>Hub_Sample_2</td><td>CountryOfInc.</td><td>-0.0069</td><td>0.0525</td><td></td><td></td></tr><tr><td>Low_Sample_2</td><td>CountryOfInc.</td><td>0.0612</td><td>-0.0012</td><td></td><td></td></tr><tr><td>Low_Sample_3</td><td>CountryOfInc.</td><td>-0.0078</td><td>0.0190</td><td></td><td></td></tr><tr><td>Mean</td><td>一</td><td>-0.0153</td><td>-0.0203</td><td>35 total</td><td>8 total</td></tr></table>

Table 8: Secondary per-report similarity audit. The earlier narrow audit also finds negligible correlation between lexical proximity and flip likelihood.

Table 10 shows that Popular-source updates produce larger perturbations at d = 1 in all three models, which is the consistent pattern summarized in the main text. Later-hop comparisons are mixed: Rare-source updates are larger at most distant hops for Qwen3.5-2B, while the ordering varies by hop for Qwen3.5-9B and Gemma-4-31B-it. We therefore treat attention perturbation as a diagnostic associated with the immediate-neighbor pattern, not as evidence of a universal causal mechanism across all distances.

<table><tr><td>Model</td><td>Source</td><td>d1</td><td>d2</td><td>d3</td><td>d4</td><td>d5</td></tr><tr><td>Qwen3.5-2B</td><td>Popular Rare</td><td>0.956</td><td>0.476</td><td>0.468</td><td>0.534</td><td>0.524</td></tr><tr><td></td><td>Popular</td><td>0.851 0.421</td><td>0.754</td><td>0.584</td><td>0.564</td><td>0.597</td></tr><tr><td>Qwen3.5-9B</td><td>Rare</td><td>0.187</td><td>0.334 0.311</td><td>0.284 0.274</td><td>0.277 0.255</td><td>0.242 0.273</td></tr><tr><td>Gemma-4-31B-it</td><td>Popular</td><td>0.139</td><td>0.083</td><td>0.096</td><td>0.106</td><td>0.102</td></tr><tr><td></td><td>Rare</td><td>0.102</td><td>0.093</td><td>0.076</td><td>0.071</td><td>0.076</td></tr></table>

Table 10: Hop-wise attention perturbation in the three completed paired audits. Each cell reports mean |∆AttLift| on the clean-correct subset; bold marks the larger source class within each model and hop. Popularsource updates produce the larger perturbation at d=1 in all three models, whereas later-hop comparisons are mixed.