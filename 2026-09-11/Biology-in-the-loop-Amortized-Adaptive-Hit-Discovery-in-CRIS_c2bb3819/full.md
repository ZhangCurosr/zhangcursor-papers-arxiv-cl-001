# Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens

Carl EdwardsEdward De BrouwerXiner Li

Namkyeong Lee Ehsan Hajiramezanali Anne Biton Sara Mostafavi Gabriele Scalia

Genentech, South San Francisco, CA, USA

{edwarc24,debroue1,lix361,scaliag}@gene.com These authors contributed equally

## Abstract

Many biological discovery problems require experiments to be selected sequentially under constrained budgets. CRISPR screening is a prominent example, as exhaustive perturbation testing is often infeasible and candidate perturbations must instead be prioritized over multiple experimental rounds. Despite the importance of this problem, existing benchmarks for adaptive hit discovery remain limited in scale and diversity. Here, we introduce AsSAYBENCH-LOOP, a large-scale benchmark for adaptive hit discovery comprising 1,389 CRISPR screens across five phenotype categories. Beyond enabling systematic evaluation, its scale makes it possible to learn acquisition strategies across historical experiments. Building on this resource, we introduce AsSAYLoOP, a sequential experimental design framework combining AsSAYFoRMER, a transformer-based amortized acquisition policy trained across historical screens to adapt from experimental feedback, with LLM-derived biological priors through an adaptive handoff. In this view, completed experiments become training data for learning how accumulated evidence should guide what to test next, while LLMs provide prior biological knowledge to seed the search. We further introduce AsSAYLLM, showing that the same principle can be extended directly to an LLM through task-specific post-training. On temporally held-out screens, AsSAYLoOP achieves a 5.67-fold enrichment over random selection and recovers 27.7% of hits after assaying approximately 5% of the candidate library, outperforming existing adaptive-design methods and standalone LLMs, and AsSAYFORMER alone. Performance improves with increasing historical training data and transfers to phenotype categories excluded from training. These results demonstrate the value of learning acquisition policies across historical experiments and combining them with broad biological priors for efficient adaptive hit discovery.

Website

GitHub

PyPI

Models & Data

![](images/9ed68d71e00fad9d4ce4efa1328a33944924b7396981eb3dc2822fa7f7a775cf.jpg)  
From historical CRISPR screens to a sequential experimental-design benchmark  
Figure 1: Overview of AsSAYBENCH-LOOP and AsSAYLOOP for adaptive hit discovery in CRISPR screens.

## 1 Introduction

CRISPR screens are central tools in functional genomics and drug discovery, enabling systematic perturbation of gene function and measurement of heterogeneous phenotypic effects [4, 18, 35]. Yet the space of possible perturbations is vast, spanning thousands of genes across diverse cellular contexts, disease models, and phenotypic readouts. Exhaustive experimentation is therefore often infeasible, particularly when the biological question requires richer models or phenotypes that cannot be measured in highly multiplexed formats. In these settings, only a fraction of candidate perturbations can be tested, often over a sequence of experimental rounds [18, 4]. Efficient screen design, selecting the most informative experiments to perform next, is therefore essential for rapidly identifying biologically meaningful effects.

Sequential experimental design provides a natural framework for this setting: a model proposes experiments, observes their outcomes, and uses this feedback to determine subsequent actions under a finite budget [37]. This model-experiment interaction forms a lab-in-the-loop process, in which experimental outcomes guide the next round of decisions. Closing this loop is a core capability for emerging AI-driven and increasingly autonomous scientific systems [8, 5].

Several methods have applied this principle to perturbation-screen design. DiscoBAX proposed an objective-driven Bayesian active-learning method encouraging coverage of diverse biological mechanisms [31]. BioBO incorporates multimodal gene representations and pathway information into Bayesian optimization [27]. Probability-of-Hit fits a probabilistic surrogate to observations collected from the current screen [40]. However, these approaches treat each new experiment largely as an independent problem, fitting predictive models or acquisition strategies for each new screen. Therefore, while they can adapt to assay-specific feedback, they do not leverage the growing collection of related experiments that have already been completed.

Large language models (LLMs) have recently been explored as an alternative, serving as acquisition policies for biological experimental design [38, 20, 29]. By conditioning on natural-language descriptions of the experimental setting, LLMs can draw on broad scientific knowledge to prioritize plausible candidates before any observation has been collected. However, whether LLMs reliably adapt their acquisition strategies from experimental feedback remains an open question, with recent studies reaching conflicting conclusions [17, 51]. This reveals a central tension in adaptive biological experimentation: effective acquisition requires both strong prior knowledge for initial prioritization and a reliable mechanism for learning from assay-specific observations as they accumulate.

Despite the importance of adaptive hit discovery for functional genomics and target identification [49], existing benchmarks remain limited in scale and diversity. GeneDisco introduced evaluation protocols for active-learning policies on genetic perturbation experiments [33], but contains only four immunology-focused screens, and other studies have evaluated sequential design on similarly narrow collections [38, 20, 24]. No general-purpose, multi-phenotype benchmark at a scale that would support both rigorous evaluation and the training of transferable acquisition policies has been available. Such a resource is necessary not only to evaluate sequential-design methods, but to ask a broader question: can experimental systems learn from experience accumulated across many previous experiments?

Here, we address both the benchmarking gap and the methodological tension outlined above (Fig. 1). First, we introduce AsSAYBENCH-LOOP, a large-scale benchmark for adaptive hit discovery comprising 1,389 CRISPR screens across five phenotype categories, equipped with dedicated metrics that account for hit-rate heterogeneity and incomplete ground truth. Beyond enabling systematic evaluation, the scale of AsSAYBENCH-LoOP makes it possible to learn acquisition strategies from historical experiments rather than fitting each screen independently.

Building on this resource, we introduce AsSAYLoOP, a sequential experimental design framework that combines an amortized acquisition policy trained across historical screens with biological prior knowledge derived from an LLM. At its core is AsSAYFORMER, a transformer-based policy that learns, across historical assays, how experimental feedback should modify candidate prioritization, and transfers these learned strategies to new assays at inference time. In this view, completed experiments become training data for learning how to experiment, i.e., how evidence accumulated during a new campaign should guide what to test next. AsSAYLooP complements this learned policy with LLM-derived biological priors through an adaptive handoff: an LLM guides early acquisitions when assay-specific evidence is limited, before transitioning to AsSAYFoRMER as experimental observations accumulate. This design combines two complementary capabilities: broad biological prior knowledge for early prioritization, and a feedback-conditioned policy learned from historical experiments for subsequent adaptation.

We further explore whether the same principle, learning adaptive acquisition behavior from historical screens, can be realized directly within an LLM through task-specific post-training. To this end, we introduce AsSAYLLM, a 27B-parameter LLM fine-tuned on acquisition trajectories and subsequently optimized with reinforcement learning over sequential campaigns, providing a proof of principle for this strategy.

On temporally held-out screens, AsSAYLOOP achieves a 5.67-fold enrichment over random selection and recovers 27.7% of hits after assaying only approximately 5% of the candidate library, outperforming existing adaptive-design methods, standalone LLMs, and its core policy used alone. We further show that the learned acquisition strategy transfers to unseen phenotype categories and improves with increasing amounts of historical training data, supporting the idea that experimental decision-making can be amortized across historical experiments. Across acquired perturbations, AsSAYLooP retains broad biological pathway coverage while improving hit discovery. Analysis of the learned acquisition policy further reveals directional gene-gene influence patterns, including relationships supported by prior biological evidence.

Together, these results establish historical experimental repositories as a substrate for evaluating and learning transferable decision-making. By coupling learned policies with pretrained biological knowledge and accumulated experimental feedback, we provide a framework for improving sequential experimental design across new biological screens. More broadly, these findings suggest a path toward lab-in-the-loop systems that accumulate experimental experience across campaigns and use it to support increasingly autonomous, adaptive experimentation in biology.

## 2 Results

## 2.1 A large collection of historical CRISPR screens enables learning a transferable acquisition policy for adaptive hit discovery

We formalize adaptive hit discovery as a sequential experimental design problem (Fig. 2A). A CRISPR screen s is defined by a natural-language description $c _ { s }$ of the experimental setup, a screen library $L _ { s }$ of the genes measured in that screen, and binary hit labels $y _ { g } \in \{ 0 , 1 \}$ for each $g \in L _ { s }$ . Because a single policy must act across screens with different libraries, we additionally define a fixed candidate gene pool $\mathcal { G } _ { : }$ , shared by all screens and containing every screen library $( L _ { s } \subseteq { \mathcal { G } } )$

At each round $t \in \{ 1 , \ldots , T \}$ , the model selects a batch $B _ { t }$ of b genes, observes their hit labels, and accumulates a history $h _ { t }$ . The goal is to maximize the total number of discovered hits within the fixed budget $T \times b$ by learning an acquisition policy $\pi ( B _ { t } \mid h _ { t - 1 } )$

Asking whether completed experiments can inform future ones requires a collection of screens large enough to both train and evaluate acquisition strategies. To this end, we introduce AsSAYBENCH-LOOP, a large-scale compendium of historical CRISPR screens for adaptive hit discovery built upon AssayBench [10]. AsSAYBENCH-LOOP comprises 1,389 CRISPR screens across five phenotype categories, with a temporal train/validation/test split (1,349/20/20 screens; Fig. 2B). Validation and test screens were restricted to genomewide assays with sufficient hit signal and non-trivial baseline recoverability, and were then selected to provide broad phenotype coverage and maximize within-category diversity of screen descriptions (Fig. 2C; selection details in Appendix $\mathbf { A } ,$ phenotype composition in Supplementary Table 1 and per-screen characteristics in Supplementary Table 2). Test screens are substantially different from the training set (Fig. 2E; median test-to-training hit-set Jaccard similarity = 0.064). At the same time, AsSAYBENCH-LOOP spans a broad hit repertoire: although the most recurrent hits are enriched for DepMap common-essential genes, many non-essential genes also recur as hits across multiple screens (Fig. 2F). This structure motivates acquisition policies that can exploit recurrent cross-screen structure while retaining broad coverage of the gene space, rather than collapsing onto a small set of ubiquitous hits.

![](images/ef7062a6d91b30ad49c535f946ac75bf341d8ff41c602b82a6a5a369b1e82898.jpg)

B.  
![](images/b257937dbfe77fc5c67d1859c3065581ac622268233a7c4844b83d5ba50e3c3c.jpg)

C.  
![](images/a9876c1a61b4dc9894815019647567fe98c78a361c25f51b3ad7df28ee5585f2.jpg)

D.  
![](images/4a9f77222aa582cdb508a86e3e25c4247a761318fcf6f2cc5b7e494cc899b0aa.jpg)

E.  
![](images/11e84f04df3f5e91bd60c780ba192de2a051f76da6dc2f10fd1465b24ea55e28.jpg)

![](images/8c4f118b203e6179056766985ed4ba17487a9f2e5626e3db9675ba20a4c20cfe.jpg)

![](images/a685f439c9fd2db47e61f3854b3781d17423808cd6a3ff076532c2132fbdfb5f.jpg)

Figure 2: ASSAYBENCH-LOOP is a large-scale compendium and benchmark for adaptive hit discovery in CRISPR screens A, Problem setup. An acquisition policy receives a natural-language description $c _ { s }$ of the screen (cell line, perturbation modality, treatment and readout) and, over $T$ rounds, nominates a batch $B _ { t }$ of b genes to perturb, observes their binary hit labels, and conditions the next batch on the accumulated history. The objective is to maximize the number of hits discovered within the fixed budget $T \times b$ (here $T = 1 0$ rounds of $b = 1 0 0$ genes, i.e. 1,000 acquisitions per screen). B, Benchmark construction. Starting from AsSAYBENCH [10], screen prompts were adapted to the sequential setting and the temporally held-out validation (2021) and test (>2021) pools were filtered to 20 genome-wide screens each. The resulting AsSAYBENCH-LoOP comprises 1,389 screens split temporally into 1,349 training, 20 validation and 20 test screens. C, Phenotype composition of each split across the five phenotype categories. Training screens are dominated by fitness/proliferation assays, whereas validation and test screens were selected to spread across categories and to maximize within-category diversity of screen descriptions. D, Evaluation. Metrics are defined over the candidate gene pool ${ \mathcal { G } } ,$ the screen library $L ,$ the hit set H and the acquired genes $G .$ The primary metric is the hit enrichment factor (EF), defined as the ratio of hits found to the number expected under random selection. The normalization penalizes hallucinated and unfilled acquisitions but does not penalize valid genes absent from the retrospective screen library. Secondary metrics include the normalized area under the cumulative-hits curve $( \mathrm { n A U C } ) .$ , the fraction of hits found (FH), and the effective number of Reactome pathway groups (EP). E, Jaccard similarity between the hit set in each screen and the nearest training screen. The histogram separates within-training screens from the same or different publications; points show validation and test screens, and dashed lines mark the train-train and test medians. $\mathbf { F } ,$ Training hit frequency by gene (green; symlog scale) and the fraction of DepMap common-essential genes in a 201-gene rolling window (orange). G, Hit-rate distribution across all 1,389 screens, with validation and test screens below (each point is a screen); the dashed line marks the median.

Evaluating adaptive hit discovery requires metrics that account for substantial variation in baseline screen difficulty (Fig. 2G) and incomplete retrospective ground truth. Not all genes in $\mathcal { G }$ are measured in every screen, and LLM-based policies can hallucinate invalid gene names or return incomplete batches. To this end, our primary metric is the hit enrichment factor (EF), defined as the ratio of discovered hits to the expected number under random selection, with a normalization that penalizes hallucinated genes but not valid genes absent from the screen library (Fig. 2D; Methods). We additionally report the normalized area under the cumulative-hits curve (nAUC), the fraction of hits found (FH), and the effective number of Reactome pathway groups (EP) as a measure of biological diversity (eight metrics in total; Appendix H). All results are reported at a fixed acquisition budget per screen (10 rounds of 100 requested genes).

AsSAYBENCH-LOOP provides a substantially larger historical training resource than existing benchmarks, while its held-out evaluation sets comprise phenotype-diverse genome-wide screens. For example, GeneDisco [33] is two orders of magnitude smaller, spanning four immunology screens, and other evaluations have used similarly narrow collections [38, 20, 24]. AsSAYBENCH-LoOP provides the scale to learn a transferable acquisition strategy from historical experiments, rather than fitting each screen independently, motivating the amortized approach we describe next.

## 2.2 ASSAYFORMER learns a feedback-conditioned acquisition policy across experiments, which ASSAYLOOP combines with LLM priors

Existing sequential design methods for CRISPR screens fit a separate surrogate model for each new experiment and do not transfer acquisition strategies across screens. In contrast, we learn a shared, history-conditioned acquisition policy across all historical screens, following the amortized experimental design paradigm [13, 3, 23]. Here, the policy itself is amortized, meaning that a single set of parameters maps any screen description and accumulated experimental history to a ranking over candidate genes. Thus, the strategy for translating experimental feedback into the next action is learned from a large collection of completed experiments. The resulting policy transfers the learned screening strategy to new assays at inference time while adapting its decisions based on the perturbation-outcome pairs observed during the current experiment (Fig. 3A).

We instantiate this policy as a transformer encoder, AsSAYFORMER, that takes as input the screen description $c _ { s }$ and the history of observed gene-outcome pairs $h _ { t - 1 }$ , and produces acquisition scores over all candidate genes (Fig. 3B, Methods). Each observed gene is represented by a learnable embedding combined with a learned hit-indicator token. The screen description is encoded through a frozen text encoder and projected into the transformer's latent space. The transformer jointly processes the sequence of observed genes with the description token, and the output embedding at the description position is scored against all candidate gene embeddings via a bilinear head to obtain per-gene acquisition scores. At inference time, batches are selected greedily according to these scores. Full architectural details are provided in Methods.

Training proceeds in three stages (Fig. 3C, Methods). First, gene embeddings are initialized using Bayesian Probabilistic Matrix Factorization (BPMF) [41] on the binary hit matrix across training screens, thereby capturing cross-screen co-hit structure. Second, the model is trained with supervised learning to predict hit labels for unobserved genes using binary cross-entropy. To mimic intermediate states of a sequential screen, we randomly sample observed gene sets of varying sizes during training. Third, the policy is fine-tuned with reinforcement learning using group-relative policy optimization (GRPO) [42] on full T-step trajectory rollouts. Notably, a context-delta reward encourages the policy to leverage the experimentally revealed history by rewarding improvements over a context-free baseline (Methods).

AsSAYFORMER is explicitly trained to use experimental feedback, but its initial acquisitions rely primarily on cross-screen structure and the limited assay-specific information contained in the screen description. We therefore combine the learned policy with LLM-derived biological priors through an adaptive handoff strategy (Fig. 3D).

E. Worked-out example  
A. Active hit discovery paradigms  
B. AssayFormer  
![](images/941a658f68ade84efe376ba47b1240dab99ec02bb0b089ad9b739b278667bb49.jpg)

C. Training Assayformer  
![](images/5bd61ac5b05a5c023b7370b66e9cc9213a79e0e1243f6fb025cd9d90b245131d.jpg)

![](images/2c29cc83d23f9fb5eadc1f4dbca194e9b9a7dbe37ddb60730653ba782d075277.jpg)

![](images/2deb17f1c85047d5bf782e160c791c22b1fa93334233cbb09bdb83e2d59aa869.jpg)

![](images/1b3b4ef333fb76e2d66747623cc5125c3428e7d24552646b3e386024fe6c5446.jpg)

![](images/490b2e1e7c564ea9f52de88b0aa09d08a8f6d0f302bf5d19e41ffc424275c637.jpg)  
Figure 3: ASSAYLOOP combines ASSAYFORMER, an amortized acquisition policy for adaptive hit discovery, with LLM prior knowledge. A, Paradigms for active hit discovery. Conventional adaptive designs fit a surrogate model independently for each new screen and do not transfer acquisition strategies across experiments. LLM-based policies can exploit broad biological

knowledge from the scientific literature, but are not explicitly trained across historical screens for feedback-conditioned adaptation. In contrast, AsSAYLooP follows an amortized design paradigm: it learns a shared, history-conditioned acquisition policy across historical screens and combines it with LLM-derived priors. B, AsSAYFORMER architecture. The screen description cs is embedded by a frozen text encoder and projected into the model's latent space, while each previously assayed gene is represented by a learnable gene token combined with a learned hit-indicator token. A transformer encoder processes this unordered sequence, and the output embedding at the description position is scored against all candidate gene embeddings through a bilinear head to produce per-gene acquisition scores $a _ { \phi } ( g \mid h _ { t - 1 } , c _ { s } )$ . The next batch consists of the top-b previously untested genes. C, Three-stage training. Gene tokens are initialized by Bayesian probabilistic matrix factorization (BPMF) of the binary hit matrix of the training screens. The policy is then trained with binary cross-entropy (BCE) to predict hit labels for unobserved genes given simulated histories of varying length, and finally fine-tuned with GRPO on full T-step rollouts using a context-delta reward that credits the policy only hits recovered beyond those obtained by a context-free reference policy. D, The AsSAYLoOP handoff. An LLM acts as the acquisition policy for the first k rounds, providing a biologically informed warm start. AsSAYFORMER then takes over for the remaining T — k rounds and adapts to the accumulated experimental history. E, Worked-out example of a genome-wide CRISPR screen for regulators of TNFα-induced NF-κB activity in HeLa cells (169 true hits). (Left) Ten-round AssayFormer trajectory. Proposed genes are above the axis and recovered hits below, colored by broad Reactome category; round 4 is highlighted. (Middle) Composition of the round-4 batch with labels withheld or available as context: both batches contain 100 genes and recover 7 or 11 hits, respectively. (Right) Ranks of the union of the top 20 genes with and without labels as context. Blue and orange indicate at least twofold promotion or demotion when labels are supplied, gray indicates smaller changes, green points mark true hits. Teal annotations show leave-one-label-out counterfactual ranks.

For the first k rounds (k = 3, selected on the validation set), which we refer to as the warm-start phase, an LLM serves as the acquisition policy, processing the screen description and accumulated history in context to propose biologically informed gene selections. AsSAYLOOP then hands off to AsSAYFORMER for the remaining T — k rounds, allowing the learned policy to leverage the screen-specific feedback accumulated during the warm-start phase. This design combines two complementary capabilities: biological prior knowledge for early prioritization, provided by the LLM, and history-conditioned adaptation to subsequent experimental outcomes, learned by the trained policy.

In a representative genome-wide CRISPR screen for regulators of TNFα-induced NF-κB activity, (Fig. 3E-G), AsSAYFORMER recovered 60 of 169 true hits within the first 1,000 assayed genes. At round 4, incorporating the 24 hit labels observed among the first 300 genes substantially reranked candidate genes for the next batch, producing a net increase from seven to eleven hits and expanding their represented pathways from four to nine categories. Leave-one-label-out counterfactuals further show that specific observations, such as KCTD10 and NHLRC2, contribute strongly to the prioritization of CYFIP1, YARS2, and STRAP (these dependencies describe the model's decision process rather than established biological interactions).

## 2.3 ASSAYLOOP improves adaptive hit discovery on held-out screens

ASSAYFORMER was trained on the 1,349 historical screens in the ASSAYBENCH training set and evaluated on the test set. A broad set of methods was also benchmarked in the same setting. These spanned adaptive experimental design algorithms, transfer learning, LLMs, and agentic approaches. Adaptive methods included Probability-of-Hit [40] and BioBO [27], which fit per-screen surrogates without leveraging historical data. Transfer-based baselines included BPMF with greedy acquisition, Screen-kNN (which scores candidates using hit rates from similar training screens), MAML [12] (which meta-learns an initialization across training screens that can be quickly adapted on the observed hit labels), and a prior-hit frequency baseline. We evaluated several LLM families (GLM, Kimi, Claude, Qwen, Gemini, GPT), spanning open- and closedweight models at different sizes, and test each both with and without access to feedback labels. After each acquisition round, the accumulated experimental history is appended to the next prompt. Finally, we included two agentic harnesses specifically designed for adaptive hit discovery, LLMNN [17] and ICBR-EF [51], and a general coding agent based on Claude Haiku-4.5 with direct access to the training screens. Full details of all tested methods are provided in Methods. Table 1 summarizes test set results, with full results reported in Supplementary Table 4.

Overall, AsSAYLoOP achieves the strongest performance, with both Gemini-3.1-Pro and GPT-5.6 Sol warm starts reaching EF ≈ 5.67, nAUC 21.7%, and recovering approximately 27.7% of hits at 5% effective library coverage. This exceeds standalone Gemini-3.1-Pro (EF 4.71) and GPT-5.6 Sol (EF 4.81), as well as AsSAYFORMER alone (EF 4.83), supporting the complementarity of the LLM warm start and ASSAYFORMER feedback-conditioned policy.

Table 1: Performance comparison on held-out test screens. Metrics evaluate enrichment factor (EF; hit enrichment relative to random selection), normalized area under the cumulative-hits curve (nAUC), fraction of hits found (FH), shortfall (SF), percentage of recovered hits that are DepMap common-essential genes (%ess), and the effective number of Reactome level-2 pathway groups at batch (EP-B), screen (EP-S), and dataset (EP-D) scope. For EP-B/EP-S/EP-D, each gene is assigned to one of its pathway groups at random and each scope is subsampled to a fixed annotated-gene count (30, 200, and 6000, respectively), so the three metrics are not directly comparable across scopes. Random is an instantiation of the random policy, not an expected value. All performance metrics are computed separately for each screen and then averaged equally across the 20 test screens. See full results in Supplementary Table 4.
<table><tr><td>Method</td><td></td><td>EF (↑) nAUC (%)(↑)</td><td>FH (%)(↑)</td><td>| SF (%)(↓)</td><td>%ess (%) | EP-B</td><td></td><td>EP-S</td><td>EP-D</td></tr><tr><td>Base LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GLM-5.1</td><td>4.00</td><td>16.3</td><td>21.0</td><td>10.6</td><td>28.9</td><td>13.3</td><td>30.8</td><td>64.0</td></tr><tr><td>w/o labels</td><td>3.65</td><td>15.0</td><td>19.1</td><td>7.3</td><td>27.6</td><td>14.4</td><td>33.4</td><td>64.6</td></tr><tr><td>Gemini-3.1-Pro</td><td>4.71</td><td>20.6</td><td>24.6</td><td>3.6</td><td>30.6</td><td>14.6</td><td>34.6</td><td>66.2</td></tr><tr><td>w/o labels</td><td>4.38</td><td>18.5</td><td>22.9</td><td>3.1</td><td>29.5</td><td>14.3</td><td>34.3</td><td>63.5</td></tr><tr><td>GPT-5.6 Sol</td><td>4.81</td><td>20.5</td><td>25.2</td><td>5.0</td><td>29.6</td><td>13.2</td><td>29.5</td><td>61.8</td></tr><tr><td>AssayLLM (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.6-27B (base)</td><td>2.65</td><td>10.5</td><td>13.9</td><td>23.4</td><td>30.9</td><td>14.3</td><td>37.5</td><td>68.9</td></tr><tr><td>+ SFT (GLM-5.1 traces)</td><td>3.56</td><td>14.7</td><td>18.7</td><td>19.0</td><td>35.9</td><td>13.6</td><td>34.3</td><td>64.1</td></tr><tr><td>+ SFT + GRPO (= AssayLLM)</td><td>3.69</td><td>15.5</td><td>19.3</td><td>16.7</td><td>37.8</td><td>13.7</td><td>34.7</td><td>64.8</td></tr><tr><td>Adaptive Experimental Design Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>1.04</td><td>3.4</td><td>4.9</td><td>10.0</td><td>18.6</td><td>21.7</td><td>56.9</td><td>82.6</td></tr><tr><td>Prior hit baseline</td><td>2.87</td><td>11.5</td><td>14.9</td><td>1.5</td><td>99.1</td><td>17.8</td><td>38.4</td><td>48.4</td></tr><tr><td>Screen-kNN</td><td>3.40</td><td>12.5</td><td>17.3</td><td>3.7</td><td>77.1</td><td>18.9</td><td>42.8</td><td>54.8</td></tr><tr><td>BPMF [41]</td><td>4.49</td><td>12.3</td><td>19.7</td><td>18.0</td><td>36.1</td><td>20.1</td><td>50.7</td><td>72.5</td></tr><tr><td>MAML (w/ BPMF embs.) [12, 41]</td><td>3.77</td><td>14.3</td><td>18.6</td><td>6.4</td><td>74.9</td><td>19.4</td><td>44.4</td><td>57.8</td></tr><tr><td>BioBO [27]</td><td>2.59</td><td>5.9</td><td>12.2</td><td>10.3</td><td>38.4</td><td>19.9</td><td>50.6</td><td>72.6</td></tr><tr><td>Probability-of-hit [40]</td><td>2.41</td><td>7.3</td><td>11.8</td><td>9.0</td><td>35.2</td><td>20.3</td><td>53.4</td><td>79.0</td></tr><tr><td>Agent Harnesses</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Haiku-4.5 Agent</td><td>3.20</td><td>14.3</td><td>16.8</td><td>20.6</td><td>49.8</td><td>18.6</td><td>47.3</td><td>68.3</td></tr><tr><td>LLMNN [17]</td><td>2.39</td><td>9.4</td><td>12.5</td><td>0.1</td><td>22.6</td><td>20.3</td><td></td><td>78.6</td></tr><tr><td>ICBR-EF [51]</td><td>2.76</td><td>9.3</td><td>14.5</td><td>0.9</td><td>32.2</td><td>18.9</td><td>52.1 44.1</td><td>68.5</td></tr><tr><td>AssayFormer (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AssayFormer (random embs.)</td><td>2.47</td><td>9.2</td><td>12.3</td><td>5.4</td><td>86.1</td><td>18.5</td><td>40.0</td><td>51.5</td></tr><tr><td>+ BPMF Embeddings</td><td>3.83</td><td>13.2</td><td>18.3</td><td>10.2</td><td>37.5</td><td>18.3</td><td>42.7</td><td>56.9</td></tr><tr><td>+ GRPO (= AssayFormer)</td><td>4.83</td><td>17.2</td><td>23.2</td><td>10.0</td><td>40.7</td><td>19.3</td><td>46.7</td><td>65.0</td></tr><tr><td>AssayLoop (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GLM-5.1 → AssayFormer</td><td>5.27</td><td>19.8</td><td>25.5</td><td>8.5</td><td>31.6</td><td>18.8</td><td>47.5</td><td>71.0</td></tr><tr><td>Gemini-3.1-Pro → AssayFormer</td><td>5.67</td><td>21.7</td><td>27.7</td><td>7.6</td><td>30.1</td><td>18.4</td><td>46.6</td><td>70.4</td></tr><tr><td>GPT-5.6 Sol → AssayFormer</td><td>5.67</td><td>21.7</td><td>27.6</td><td>7.8</td><td>29.4</td><td>18.1</td><td>45.9</td><td>69.8</td></tr><tr><td>AssayLLM → AssayFormer</td><td>5.05</td><td>18.6</td><td>24.7</td><td>7.8</td><td>34.8</td><td>18.8</td><td>47.5</td><td>70.8</td></tr><tr><td>Handoff-trained AssayLLM → AssayFormer</td><td>5.23</td><td>18.7</td><td>25.1</td><td>9.1</td><td>39.2</td><td>18.9</td><td>47.0</td><td>68.3</td></tr></table>

Analyzing standalone LLM policies, we observe relatively strong performance, led by GPT-5.6 Sol, Gemini-3.1-Pro, and GLM-5.1. Removing outcome labels consistently reduces performance, indicating that LLMs use experimental feedback to inform subsequent acquisitions. However, the magnitude of the reduction is modest relative to the performance retained without labels, suggesting that their acquisition behavior remains strongly driven by information available before observing assay-specific outcomes. These results help revisit and contextualize previous findings on LLM-driven sequential experimental design [17, 51] (additional details in Appendix K), while highlighting a limitation of standalone LLM policies that AsSAYLOOP is explicitly designed to overcome.

AsSAYLOOP achieves considerably higher mean EF than methods that adapt within a screen without leveraging historical screen data, including BioBO, Probability-of-Hit, LLMNN, and ICBR-EF. The comparatively stronger performance of BPMF, Screen-kNN, and MAML, methods not specifically designed for adaptive hit discovery but that leverage available training screens, further underscores the value of transferring knowledge from historical experiments, a central motivation behind the design of AsSAYFORMER.

Figure 4A illustrates the complementary strengths of the two components of AsSAYLOOP. The LLM (Gemini 3.1 Pro shown in the figure) performs strongly in the early rounds, where its extensive biological prior gives it an advantage over methods that must learn from iterative feedback, but its performance gradually plateaus. By contrast, AsSAYFORMER relies primarily on experimental feedback: it starts from lower scores, with limited screen-specific information before the first batch, but improves steadily as outcomes accumulate. AsSAYLoOP combines these strengths, leveraging the LLM to prioritize promising hits in the initial rounds before handing off to the learned policy, which expands discovery by adapting to the accumulating experimental feedback. Supplementary Fig. 1 illustrates the same pattern in more detail for two individual test screens

## 2.4 Adaptive experimental capability scales with historical experience and transfers across phenotype categories

Having established AsSAYLOoP performance on held-out screens, we next investigated how historical screens contribute to policy learning, how far the learned policy transfers beyond the biology represented in training, and which design choices drive performance.

First, we investigated data-scaling behavior by examining how AsSAYFORMER performance varies with the number of available training screens. As shown in Figure 4C, mean performance increases across the evaluated training-set sizes, with EF@ 10 more than doubling over the scaling range and no clear evidence of saturation at the largest dataset size. By contrast, increasing model capacity yields little additional benefit across a nearly 40-fold range in parameter count, indicating that performance in this regime is primarily limited by the number of historical screens rather than model size (Appendix J).

We next evaluated transfer to phenotype categories absent from training using leave-one-phenotype-out (LOPO) models across the five phenotype categories in AsSAYBENCH-LoOP, holding out one category for evaluation while training on screens from the remaining four. The amount of data withheld varies substantially across these experiments (Supplementary Table 1), therefore we interpret LOPO primarily as a test of robustness to phenotype exclusion. Across the full test set, AsSAYFORMER trained under LOPO reaches EF 4.59, compared with 4.83 under full training, resulting in a smaller mean EF drop than Screen-kNN LOPO (0.24 versus 0.42; Fig. 4B). Overall, AsSAYFORMER under LOPO remains above other methods trained on all phenotypes, including Screen-kNN (3.40), MAML (3.77), and BPMF (4.49). These results indicate that AsSAYFORMER performance is not solely dependent on access to training screens from the same phenotype category, supporting its generalization to unseen phenotype categories.

A. Fraction of total hits found by fraction of library acquired  
![](images/797a83d4b94b210bf7e98ee2900c73cf17f4de0c5e272c36517d82ba67aa65d3.jpg)

B. Enrichment factor on all test screens  
![](images/29a5de9364ac03428bb05e34d6847635108f7b3af93708788604c2f36d112aa8.jpg)

D.  
![](images/0c7c7839a868e80dbb0ae8dd76ec7f64a6739d0515f603bc3cacb85233d69361.jpg)

C.  
![](images/0ffc6f0b34809968a1f150c7a8e401b87481a3b1bb972c9962463f4b60672be8.jpg)

![](images/83c36be3b99a0aa78eae937374d03a80a35c57a1b86031e1d2d7ef18bfed21c8.jpg)

![](images/5ed4efde5d4c43a556e59829b035ccd2e0a549c4ca5dadf9c8e35ebc906df327.jpg)  
Figure: 4: AsSAYLOoP outperforms existing methods at adaptive hit discovery and improves with historical experience All panels report performance on the 20 temporally held-out test screens of AsSAYBENCH-LOOP (T = 10 rounds of b = 100 genes). A, Fraction of all screen hits recovered as a function of the effective fraction of the library acquired, averaged over test screens. AsSAYLoOP recovers hits fastest throughout the campaign. Gemini 3.1 Pro is competitive over the first rounds, reflecting a strong biological prior, but its trajectory flattens as observations accumulate, whereas AsSAYLOOP hands off to AsSAYFORMER and continues to gain. B, Phenotype-held-out transfer. EF for AsSAYFORMER and Screen-kNN trained on the full training set, and in a leave-one-phenotype-out (LOPO) regime in which every training screen belonging to the evaluated phenotype category is removed. Screen-kNN degrades sharply under LOPO, while AsSAYFORMER is largely preserved and, under LOPO, still exceeds Screen-kNN trained on all phenotypes. C, Data scaling. 216 AsSAYFORMER models were trained on nested subsets of 1 to 1,349 training screens (more models were trained in the low-data regime to reduce noise). Left, EF at acquisition step k; right, the acquisition budget (as a percentage of the library) required to reach a given recall of the screen hits. Solid lines, supervised fine-tuning; dashed lines, after GRPO fine-tuning; shading, the gain attributable to RL. Performance increases monotonically with the number of training screens, with no clear plateau over the evaluated range. For this scaling analysis, we omit the screen-description input to isolate scaling trends of the in-context (i.e., history-conditioned) capabilities. D, Gene token initialization. EF after SFT (light) and after RL fine-tuning (dark) for gene embeddings initialized at random, from GenePT, from K562 expression, and from three factorizations of the training hit matrix (SVD, MF, BPMF). Initializations that encode cross-screen hit structure substantially outperform embeddings derived from external biological data. E, AsSAYFORMER ablations. Removing the screen description $c _ { s }$ from the input, or replacing the context-delta reward with EF directly, both reduce EF

Finally, we performed multiple ablations to identify important architectural choices (Fig. 4D-E, full ablations in Supplementary Table 3). Two design choices contributed meaningfully to performance and offered additional insight into how the policy operates. The first aspect is gene token initialization: in our experiments, embeddings derived from the training hit matrix (BPMF, SVD, MF) outperform those derived from external biological data, such as GenePT [7] or K562 essential-gene Perturb-seq experiments [28], with BPMF resulting in the best performance after RL fine-tuning (EF 4.83 vs. 4.40 for SVD; Fig. 4D). The second aspect is that policy training benefits from both conditioning on the assay and explicitly rewarding the policy for exploiting that context: either removing the screen description $c _ { s }$ or replacing the context-delta reward (which explicitly encourages leveraging the context) with terminal EF degrades performance (Fig. 4E). While conditioning on the screen description generally improves model performance, we also notice that AsSAYLooP's handoff design provides a broader mechanism for incorporating screen-specific contextual knowledge through the LLM in the early rounds, which leads to a substantial boost in performance (Table 1).

![](images/4483f8c782489038199c83630e0549d691205977303e0ff4d8a1a7d9f449ff8a.jpg)

![](images/96a42c3761e39cef7555e21699bfb65be5327b8772397f202e3299c8af8faef6.jpg)

![](images/e5a2afacc63b60d7a92cd334ef8dbad49c4f979fc48b3ab2f8be2095d44034df.jpg)

C. Context-dependent interactions with literature-supported biological connections  
![](images/e6c56659fdde0186e3e9d95fc3cf9c4005edef322ed838f7a65eb5207121a63f.jpg)  
Figure 5: ASSAYLOOP acquires biologically diverse perturbations and encodes directional gene-gene relationships. A. Reactome pathway composition of the genes acquired by six representative methods, pooled over the 20 test screens (1,000 acquisitions per screen). Inner ring, the eight largest top-level Reactome categories, with all remaining roots pooled into Other; outer ring, second-tier groups within each category, shown as tints of the parent hue. Random is the composition of the annotated gene universe, i.e. the expected composition of a uniformly drawn batch (EP metrics are calculated for the expected distribution). The number at the centre is the dataset-level effective number of pathway groups (EP-D), values below each ring give the corresponding batch-level (EP-B) and screen-level (EP-S) quantities. Screen-kNN is the most concentrated repertoire (EP-D 54.8), over-weighting RNA metabolism; Gemini-3.1-Pro retains broad dataset-level breadth (EP-D 66.2) but has lower diversity within individual batches (EP-B 14.5); AsSAYLOOP combines high dataset-level diversity (EP-D 70.4) with broad batch coverage (EP-B 18.4). B, Directional gene-gene influence learned by AsSAYFORMER. Influence is defined as the change in the acquisition score of a target gene when a probe gene is added to the observed history as a hit, relative to a background context of 50 randomly drawn genes, averaged over 10 independently sampled backgrounds. Influence patterns are calculated under a fixed, generic screen description to inspect model-wide influence patterns learned by the model. Left, influence matrix for a panel of 31 genes comprising 12 canonical cancer

drivers (upper-left block) and 19 genes from functional modules; rows are probe genes and columns are target genes. The matrix is markedly asymmetric, showing that the model encodes directional updates rather than gene similarity. Right, the highest-magnitude probe→target pairs remaining after removal of pairs annotated in databases (Methods). C, Context-dependent influences with literature-supported biological connections. Bars compare the influence of each probe-target pair in the indicated focal screen with its mean influence across the other 19 held-out screens, holding all other inputs fixed. Both genes are true hits in the focal screen, and the pair is absent from databases. Cartoons summarize literature-supported shared biological contexts involving HUSH recruitment, translational stress sensing, and TFIIC/INO80 activity at tRNA genes.

## 2.5 ASSAYLOOP preserves biological breadth while enriching for hits

Beyond hit discovery performance, we examined the biological diversity of acquired perturbations using the effective number of Reactome pathway groups (EP) at three scopes: within a batch (EP-B), within a screen (EP-S), and across the full test dataset (EP-D).

As shown in Table 1 and Figure 5A, standalone LLMs show lower diversity within batches and screens (EP-B 13–15, EP-S 30–35), but substantially broader coverage across the full dataset. For example, Gemini-3.1-Pro reaches EP-D 66.2, close to BPMF (72.5) and BioBO (72.6). In contrast, it achieves a proportionally much lower within-screen diversity (EP-S 34.6) compared to those two methods (50.7 and 50.6). Such behavior is consistent with LLMs focusing on different screen-specific biological programs across assays, while exploring a narrower biological neighborhood within each individual screen. Such focused prioritization is well suited to providing an initial warm start, but may limit exploration over multiple acquisition rounds. Further, we find that methods that do not leverage training screens, such as BioBO and Probability-of-Hit, maintain high diversity within screens and across the dataset, consistent with broader exploration rather than concentration on historically frequent hits.

AsSAYFORMER similarly explores more broadly within individual batches and screens (EP-B 19.3; EP-S 46.7), while maintaining high dataset-level diversity (EP-D 65.0). AsSAYLOOP retains comparable breadth while improving hit-discovery performance: with Gemini-3.1-Pro, it achieves similar or higher batch-, screen and dataset-level diversity (EP-B 18.4 vs. 19.3; EP-S 46.6 vs. 46.7; EP-D 70.4 vs. 65.0) while increasing EF from 4.83 to 5.67. Thus, AsSAYLOOP combines the focused early prioritization of the LLM with AsSAYFORMER biological breadth across the screen, while achieving higher hit enrichment than either component alone. Additional per-model diversity analysis, including pathway composition across LLM families, is provided in Appendix G.

## 2.6 ASSAYFORMER encodes directional gene-gene influence patterns

Because AsSAYFORMER updates its acquisition scores as observations accumulate, we can ask how observing one gene as a hit changes the prioritization of others. We define the influence of a probe (that is, observed) gene on a target gene as the change in the target gene's acquisition score when the probe is added to an otherwise matched history as a hit, measured against a background context of 50 randomly drawn genes and averaged over 10 independently sampled contexts (Methods). The influence therefore captures a directional update encoded by the learned acquisition policy, rather than a calibrated interaction probability or evidence of causality. Nonetheless, these learned dependencies provide a way to interrogate the biological structure underlying the learned policy and to nominate gene relationships for further investigation.

First, we probed model-wide influence patterns under a fixed generic screen description, averaging over randomized background histories (Methods). Figure 5B (left) shows the influence matrix for a panel of 31 genes: 12 canonical cancer drivers and 19 genes drawn from functional modules. Within this panel, the learned relationships are asymmetric: 44% of reciprocal pairs have influences of opposite sign. For example, adding MDM2 to the history as a hit increases the predicted score of PFDN4 by 0.33, whereas adding PFDN4 decreases the predicted score of MDM2 by 0.46. These asymmetric relationships cannot be explained by a purely symmetric gene-similarity representation: observing one gene can change the prioritization of another

in a direction that is not reciprocated.

To ask whether these learned relationships extend beyond established annotations, we removed pairs represented in STRING, CORUM, SIGNOR, MSigDB, or Reactome [46, 15, 30, 45, 34] and examined the highest-magnitude remaining influences (Fig. 5B, right). Among the positive pairs, including MYC in the hit history increases the score of SMU1 and PRPF4. These learned associations are consistent with established links between oncogenic MYC and dependence on spliceosome machinery [22, 25, 9]. Similarly, MDM2 increases the prioritization of EMG1 and PNO1, two factors essential for small-subunit ribosome biogenesis. Biologically, disrupting ribosome biogenesis stabilizes p53 via the RPL5/RPL11-5S-RNP-MDM2 nucleolar surveillance pathway [19, 6]. Supporting a functional connection, PNO1 depletion has been shown to increase RPL11–MDM2 association and stabilize p53 in TP53-wild-type colorectal cancer cells [43]. Among negative influences, several high-magnitude pairs have plausible pathway-level precedent but lack direct experimental validation. For example, SMAD4 decreases the prioritization of NDUFS7 and NFU1, consistent with reported links between SMAD4 loss and altered mitochondrial respiration in pancreatic cancer [11]. Similarly, the NRAS-to-AMOTL2 influence is consistent with indirect pathway-level evidence: although oncogenic NRAS activates Hippo signaling in melanocytes, and parallel BRAF-driven MAPK activation reduces expression of the YAP/TAZ target AMOTL2 [50], AMOTL2 was not directly evaluated under NRAS mutation.

Because AsSAYFORMER conditions jointly on the screen description and observed history, we next explore whether learned influences vary across assay contexts. We search for pairs with substantially stronger influence in one held-out screen than across the remaining 19, requiring both genes to be hits in the focal screen and excluding pairs annotated in databases (Methods). For input context, we use the first 100 genes suggested by the LLM (Gemini-3.1), following the AsSAYLoOP protocol, and hit labels assigned from the ground truth data. This exploratory analysis identified three examples with literature-supported pathway-level connections (Fig. 5C): DCAF1→ZNF638, IGHMBP2→GCN1, and GTF3C1→ACTR5. In each case, influence is substantially larger in the focal screen than across other test contexts. These examples illustrate context dependence in the learned policy and nominate hypotheses for follow-up, rather than establishing direct genetic interactions.

## 2.7 Post-training LLMs can improve sequential hit discovery

Our preceding results show that historical experiments can be used to learn a feedback-conditioned acquisition policy that generalizes to held-out screens, and that this capability complements the biological knowledge encoded in pretrained LLMs. We therefore asked whether adaptive acquisition behavior could be learned directly within an LLM through task-specific post-training, thereby directly integrating learned experimental decision-making with pretrained biological knowledge.

As a proof of concept, we introduce AsSAYLLM, a domain-specific LLM acquisition policy built on the 27B-parameter Qwen 3.6 backbone. AsSAYLLM is trained in two stages (Fig. 6A): supervised fine-tuning (SFT) on complete acquisition trajectories generated by a stronger teacher model (GLM-5.1), followed by group-relative policy optimization (GRPO) to directly optimize screen-level hit discovery, with rollouts from the same screen forming a group to control for screen difficulty (Methods).

Post-training substantially improves performance on AsSAYBENCH-LOOP (Fig. 6B; Table 1). Starting from Qwen3.6-27B, SFT increases EF from 2.65 to 3.56, and subsequent GRPO further increases EF to 3.69 corresponding to a 39% improvement over the base model. AsSAYLLM approaches its larger GLM-5.1 teacher (EF 4.0), showing that a substantial fraction of the teacher's acquisition performance can be transferred to a smaller open-weight model. The post-training improvement from EF 2.65 to 3.69 substantially narrows the gap to AsSAYFORMER (4.83). These gains, combined with the strong zero-shot acquisition performance of frontier LLMs (Table 1), motivate extending the same strategy to stronger pretrained LLMs as a path toward more capable standalone adaptive policies.

We further asked whether post-training could optimize an LLM specifically for its role as the warm-start policy in AsSAYLOOP. We therefore trained a second AsSAYLLM variant in the handoff setting, in which the LLM controls the first k rounds before handing the accumulated context to AsSAYFORMER (Methods). In this setting, the objective rewards not only overall hit discovery but also the quality of the experimental history constructed for AsSAYFORMER (Methods). Using AsSAYLLM as the warm-start policy yields an EF of 5.05, while handoff-aligned training further increases EF to 5.23, approaching the GLM-5.1 → AsSAYFORMER configuration (EF 5.27). Together, these results provide a proof of principle that historical screens and simulated trajectories can be used to post-train LLMs both as standalone adaptive policies and to better complement a specialized feedback-conditioned acquisition model.

A. AssayLLM  
![](images/e6c9060f66e915a9dd7e4b46c4344bfbd0b62cd909e5cba226131505415e89f6.jpg)

![](images/5ff515770aba08814907f74a78b1ab21fa57c811f2ec76da846c4bdae75c4f84.jpg)  
Figure 6: Post-training improves sequential hit-discovery performance in an open-weight LLM. A, AsSAYLLM training pipeline. Acquisition trajectories are first collected by running GLM-5.1 as an acquisition policy over the AsSAYBENCH-LoOP training screens. A 27B-parameter Qwen3.6 backbone is then supervised fine-tuned (SFT) on these trajectories, exposing it to the sequential acquisition protocol and experimental feedback while training it to emit parseable gene batches. Two variants are optimized with group-relative policy optimization (GRPO): a standalone policy trained on full T-round rollouts (3A), and a warm-start policy trained on the handoff campaign, in which AsSAYLLM controls the first k = 3 rounds before the frozen AsSAYFORMER completes the remaining rounds (3B). B, Enrichment factor (EF) on the 20 test screens for the Qwen 3.6 27B base model, after SFT on GLM-5.1 traces, and after SFT+GRPO (AsSAYLLM), each acting as a standalone policy; and for the two AsSAYLOOP configurations in which AsSAYLLM provides the warm start before handing off to AsSAYFORMER. Post-training raises standalone EF by 39% over the base model, and the handoff-trained variant approaches the GLM-5.1→AsSAYFORMER configuration.

## 3 Discussion

We introduced AsSAYBENCH-LOOP, a comprehensive benchmark for adaptive hit discovery in CRISPR screens, spanning 1,389 screens across five phenotype categories with dedicated metrics that account for incomplete libraries and LLM-specific failure modes. By providing standardized evaluation at a scale two orders of magnitude larger than previous benchmarks, AsSAYBENCH-LOOP enables both rigorous comparison of acquisition strategies and, crucially, the training of policies that transfer across experiments.

Building on this resource, AsSAYLOOP achieves the strongest performance among the methods evaluated, recovering an average 27.7% of hits while assaying 5% of the candidate library. The handoff design in which an LLM provides biologically informed prioritization in early rounds before transitioning to AsSAYFORMER for feedback-conditioned adaptation, outperforms both components used in isolation, supporting the complementarity of these two capabilities.

Our results clarify the role of LLMs in adaptive experimental design. While we confirm that frontier LLMs do incorporate experimental feedback to some degree, their in-context adaptation remains limited compared to a policy explicitly trained for this purpose. This finding, consistent across multiple families, suggests that current general-purpose LLMs are particularly effective as sources of biological prior knowledge, while policies explicitly trained for feedback-conditioned adaptation make more effective use of accumulating experimental outcomes. Domain-specific post-training, as demonstrated by AsSAYLLM, can substantially narrow this gap and improve adaptive acquisition behavior directly within an LLM: trajectory-level supervision and deployment-aligned reinforcement learning yield a 39% improvement over the base model, and handoff-aligned training further improves its performance as the warm-start policy in AsSAYLoOP.

The data scaling analysis shows that AsSAYFORMER performance improves consistently with the number of available training screens, with no clear plateau over the range evaluated. Combined with the strong leave-one-phenotype-out generalization results, this suggests that larger and more diverse screen collections may further improve transferable acquisition policies.

Several limitations define the scope of this study. AsSAYBENCH-LOOP is a retrospective benchmark focused on genome-wide screens with sufficient hit signal, and the policies evaluated here remain to be validated prospectively in a live experimental loop. We also focus on binary hit discovery under a fixed acquisition budget; extending the framework to richer readouts and heterogeneous experimental costs will be important directions for future work. Finally, the retrospective formulation treats observed screen outcomes as fixed labels and does not explicitly model experimental noise across replicates.

Because AsSAYBENCH-LOOP represents each experimental task through a natural-language description, the same benchmark formulation can in principle extend beyond the phenotypes and modalities currently represented to encompass diverse perturbation experiments and richer biological readouts. AsSAYLLM represents a particularly promising avenue in this direction: as an LLM-based policy, it can flexibly represent new experimental readouts, perturbation types, and biological contexts through its text interface, while its post-training strategy provides a mechanism for incorporating new screening data as it becomes available. More broadly, the iterative, lab-in-the-loop paradigm explored here motivates a future in which adaptive experimental design and biological foundation model training are coupled, with each experimental campaign both advancing biological discovery and generating data that improves the models guiding subsequent experiments.

## References

[1] James H Albert and Siddhartha Chib. Bayesian analysis of binary and polychotomous response data. Journal of the American statistical Association, 88(422):669–679, 1993.

[2] Anthropic. Introducing claude opus 4.8, May 2026. URL https: //www.anthropic.com/ news/claude-opus-4-8.

[3] Tom Blau, Edwin V Bonilla, Iadine Chades, and Amir Dezfouli. Optimizing sequential experimental design with deep reinforcement learning. In International Conference on Machine Learning, pages 2107–2128. PMLR, 2022.

[4] Christoph Bock, Paul Datlinger, Florence Chardon, Matthew A Coelho, Matthew B Dong, Keith A Lawson, Tian Lu, Laetitia Maroc, Thomas M Norman, Bicna Song, Geoff Stanley, Sidi Chen, Mathew Garnett, Wei Li, Jason Moffat, Lei S Qi, Rebecca S Shapiro, Jay Shendure, Jonathan S Weissman, and Xiaowei Zhuang. High-content CRISPR screening. Nat. Rev. Methods Primers, 2(1):8, February 2022.

[5] Richard B Canty and Milad Abolhasani. The past, present and future of self-driving laboratories. Nature Reviews Chemistry, 10(8):523–537, 2026.

[6] Nestor Miguel Castillo Duque de Estrada, Matthias Thoms, Dirk Flemming, Henrik M Hammaren, Robert Buschauer, Michael Ameismeier, Jochen Baßler, Martin Beck, Roland Beckmann, and Ed Hurt. Structure of nascent 5s rnps at the crossroad between ribosome assembly and mdm2–p53 pathways. Nature structural & molecular biology, 30(8):1119–1131, 2023.

[7] Yiqun Chen and James Zou. GenePT: A simple but effective foundation model for genes and cells built from ChatGPT. bioRxiv, March 2024.

[8] Wenduo Cheng, Mingqian Ma, Shuaike Shen, Anna Hupalowska, Jennifer E. Rood, Yang Zhang, Gaurav Agrawal, Christine Bakan, Michelle A. Lee, Aviv Regev, and Jian Ma. Towards human-led, agent-driven autonomous laboratories for the life sciences. Preprints, August 2026. doi: 10.20944/preprints202608. 0273.v1.URLhttps://doi.org/10.20944/preprints202608.0273.v1.

[9] Maciej Cieśla, Phuong Cao Thi Ngoc, Eugenia Cordero, Álvaro Sejas Martinez, Mikkel Morsing, Sowndarya Muthukumar, Giulia Beneventi, Magdalena Madej, Roberto Munita, Terese Jönsson, et al. Oncogenic translation directs spliceosome dynamics revealing an integral role for sf3a3 in breast cancer. Molecular cell, 81(7):1453–1468, 2021.

[10] Edward De Brouwer, Carl Edwards, Alexander Wu, Jenna Collier, Graham Heimberg, Xiner Li, Meena Subramaniam, Ehsan Hajiramezanali, David Richmond, Jan-Christian Hütter, et al. Assaybench: An assay-level virtual cell benchmark for llms and agents. arXiv preprint arXiv:2605.10876, 2026.

[11] Zuzana Ezrova, Zuzana Nahacka, Jan Stursa, Lukas Werner, Erik Vlcak, Petra Kralova Viziova, et al. SMAD4 loss limits the vulnerability of pancreatic cancer cells to complex I inhibition via promotion of mitophagy. Oncogene, 40(14):2539–2552, 2021.

[12] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

[13] Adam Foster, Desi R Ivanova, Ilyas Malik, and Tom Rainforth. Deep adaptive design: Amortizing sequential bayesian experimental design. In International Conference on Machine Learning, pages 3384–3395. PMLR, July 2021.

[14] Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. arXiv preprint arXiv:2210.02410, 2022.

[15] Madalina Giurgiu, Julian Reinhard, Barbara Brauner, Irmtraud Dunger-Kaltenbach, Gisela Fobo, Goar Frishman, Corinna Montrone, and Andreas Ruepp. Corum: the comprehensive resource of mammalian protein complexes—2019. Nucleic acids research, 47(D1):D559–D563, 2019.

[16] Google. Gemini 3.1 pro: A smarter model for your most complex tasks, Feb 2026. URL https://blog.google/innovation-and-ai/models-and-research/ gemini-models/gemini-3-1-pro/.

[17] Rushil Gupta, Jason Hartford, and Bang Liu. LLMs for Bayesian optimization in scientific domains: Are we there yet? In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 15482–15510. Association for Computational Linguistics, 2025.

[18] Ruth E Hanna and John G Doench. Design and analysis of CRISPR-cas experiments. Nat. Biotechnol., 38(7):813–823, July 2020.

[19] Katherine M Hannan, Priscilla Soo, Mei S Wong, Justine K Lee, Nadine Hein, Perlita Poh, Kira D Wysoke, Tobias D Williams, Christian Montellese, Lorey K Smith, et al. Nuclear stabilization of p53 requires a functional nucleolar surveillance pathway. Cell reports, 41(5), 2022.

[20] Minsheng Hao, Yongju Lee, Hanchen Wang, Gabriele Scalia, and Aviv Regev. PerTurboAgent: An LLMbased agent for designing iterative perturb-seq experiments. In Machine Learning in Computational Biology, pages 44–64. PMLR, 2025.

[21] Mark O Hill. Diversity and evenness: a unifying notation and its consequences. Ecology, 54(2):427–432, 1973.

[22] Tiffany Y-T Hsu, Lukas M Simon, Nicholas J Neill, Richard Marcotte, Azin Sayad, Christopher S Bland, Gloria V Echeverria, Tingting Sun, Sarah J Kurley, Siddhartha Tyagi, et al. The spliceosome is a therapeutic vulnerability in myc-driven cancer. Nature, 525(7569):384–388, 2015.

[23] Daolang Huang, Yujia Guo, Luigi Acerbi, and Samuel Kaski. Amortized bayesian experimental design for decision-making. Advances in Neural Information Processing Systems, 37:109460–109486, 2024.

[24] Kexin Huang, Romain Lopez, Jan-Christian Hütter, Takamasa Kudo, Antonio Rios, and Aviv Regev. Sequential optimal experimental design of perturbation screens guided by multi-modal priors. In Research in Computational Molecular Biology, pages 17–37. Springer Nature Switzerland, 2024.

[25] Cheryl M Koh, Marco Bezzi, Diana HP Low, Wei Xia Ang, Shun Xie Teo, Florence PH Gay, Muthafar Al-Haddawi, Soo Yong Tan, Motomi Osato, Arianna Sabò, et al. Myc regulates the core pre-mrna splicing machinery as an essential step in lymphomagenesis. Nature, 523(7558):96–100, 2015.

[26] Wouter Kool, Herke Van Hoof, and Max Welling. Stochastic beams and where to find them: The gumbel-top-k trick for sampling sequences without replacement. In International conference on machine learning, pages 3499–3508. PMLR, 2019.

[27] Yanke Li, Tianyu Cui, Tommaso Mansi, Mangal Prakash, and Rui Liao. Biobo: Biology-informed bayesian optimization for perturbation design. arXiv preprint arXiv:2509.19988, 2025.

[28] Russell Littman, Jacob Levine, Sepideh Maleki, Yongju Lee, Vladimir Ermakov, Lin Qiu, Alexander Wu, Kexin Huang, Romain Lopez, Gabriele Scalia, Tommaso Biancalani, David Richmond, Aviv Regev, and Jan-Christian Hütter. Gene-embedding-based prediction and functional evaluation of perturbation expression responses with presage. bioRxiv, 2025. doi: 10.1101/2025.06.03.657653.

[29] Tennison Liu, Nicolás Astorga, Nabeel Seedat, and Mihaela van der Schaar. Large language models to enhance bayesian optimization. arXiv preprint arXiv:2402.03921, 2024.

[30] Prisca Lo Surdo, Marta Iannuccelli, Silvia Contino, Luisa Castagnoli, Luana Licata, Gianni Cesareni, and Livia Perfetto. Signor 3.0, the signaling network open resource 3.0: 2022 update. Nucleic acids research, 51(D1):D631–D637, 2023.

[31] Clare Lyle, Arash Mehrjou, Pascal Notin, Andrew Jesson, Stefan Bauer, Yarin Gal, and Patrick Schwab. Discobax: Discovery of optimal intervention sets in genomic experiment design. In International Conference on Machine Learning, pages 23170–23189. PMLR, 2023.

[32] Leland McInnes, John Healy, and Steve Astels. hdbscan: Hierarchical density based clustering. The Journal of Open Source Software, 2(11), mar 2017. doi: 10.21105/joss.00205. URL ht tps : //doi.org/10.21105%2Fjoss.00205.

[33] Arash Mehrjou, Ashkan Soleymani, Andrew Jesson, Pascal Notin, Yarin Gal, Stefan Bauer, and Patrick Schwab. Genedisco: A benchmark for experimental design in drug discovery. In International Conference on Learning Representations, 2022.

[34] Marija Milacic, Deidre Beavers, Patrick Conley, Chuqiao Gong, Marc Gillespie, Johannes Griss, Robin Haw, Bijay Jassal, Lisa Matthews, Bruce May, et al. The reactome pathway knowledgebase 2024. Nucleic acids research, 52(D1):D672–D678, 2024.

[35] Laralynne Przybyla and Luke A Gilbert. A new era in functional genomics screens. Nat. Rev. Genet., 23(2):89–103, February 2022.

[36] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL ht tps : //qwen.ai/blog?id=qwen3.6-27b.

[37] Tom Rainforth, Adam Foster, Desi R Ivanova, and Freddie Bickford Smith. Modern bayesian experimental design. Stat. Sci., 39(1):100–114, February 2024.

[38] Yusuf H Roohani, Andrew H. Lee, Qian Huang, Jian Vora, Zachary Steinhart, Kexin Huang, Alexander Marson, Percy Liang, and Jure Leskovec. Biodiscoveryagent: An AI agent for designing genetic perturbation experiments. In The Thirteenth International Conference on Learning Representations, 2025.URL https://openreview.net/forum?id=HAwZGLcye3.

[39] Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 627–635. JMLR Workshop and Conference Proceedings, June 2011.

[40] Andrea Rubbi, Arpit Merchant, Samuel Ogden, Amir Akbarnejad, Pietro Lio, Sattar Vakili, and Mohammad Lotfollahi. Many needles in a haystack: Active hit discovery for perturbation experiments. In Forty-third International Conference on Machine Learning, 2026.

[41] Ruslan Salakhutdinov and Andriy Mnih. Bayesian probabilistic matrix factorization using markov chain monte carlo. In Proceedings of the 25th international conference on Machine learning - ICML '08, New York, New York, USA, 2008. ACM Press.

[42] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[43] Aling Shen, Youqin Chen, Liya Liu, Yue Huang, Hongwei Chen, Fei Qi, Jiumao Lin, Zhiqing Shen, Xiangyan Wu, Meizhu Wu, et al. Ebf1-mediated upregulation of ribosome assembly factor pno1 contributes to cancer progression by negatively regulating the p53 signaling pathway. Cancer research, 79(9):2257–2270, 2019.

[44] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[45] Aravind Subramanian, Pablo Tamayo, Vamsi K Mootha, Sayan Mukherjee, Benjamin L Ebert, Michael A Gillette, Amanda Paulovich, Scott L Pomeroy, Todd R Golub, Eric S Lander, et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. Proceedings of the national academy of sciences, 102(43):15545–15550, 2005.

[46] Damian Szklarczyk, Rebecca Kirsch, Mikaela Koutrouli, Katerina Nastou, Farrokh Mehryary, Radja Hachilif, Annika L Gable, Tao Fang, Nadezhda T Doncheva, Sampo Pyysalo, et al. The string database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest. Nucleic acids research, 51(D1):D638–D646, 2023.

[47] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Ziwei Chai, Y Charles, HS Che, Cheng Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[48] Aviad Tsherniak, Francisca Vazquez, Phil G Montgomery, Barbara A Weir, Gregory Kryukov, Glenn S Cowley, Stanley Gill, William F Harrington, Sasha Pantel, John M Krill-Burger, Robin M Meyers, Levi Ali, Amy Goodale, Yenarae Lee, Guozhi Jiang, Jessica Hsiao, William F J Gerath, Sara Howell, Erin Merkel, Mahmoud Ghandi, Levi A Garraway, David E Root, Todd R Golub, Jesse S Boehm, and William C Hahn. Defining a cancer dependency map. Cell, 170(3):564–576.e16, July 2017.

[49] Bram Van de Sande, Joon Sang Lee, Euphemia Mutasa-Gottgens, Bart Naughton, Wendi Bacon, Jonathan Manning, Yong Wang, Jack Pollard, Melissa Mendez, Jon Hill, et al. Applications of singlecell rna sequencing in drug discovery and development. Nature reviews Drug discovery, 22(6):496–520, 2023.

[50] Marc A Vittoria, Nathan Kingston, Kristyna Kotynkova, Eric Xia, Rui Hong, Lee Huang, Shayna McDonald, Andrew Tilston-Lunel, Revati Darp, Joshua D Campbell, et al. Inactivation of the hippo tumor suppressor pathway promotes melanoma. Nature Communications, 13(1):3732, 2022.

[51] Gilles Wainrib, Barbara Bodinier, Haitem Dakhli, Josep Monserrat, Almudena Espin Perez, Sabrina Carpentier, Roberta Codato, and John Klein. Can ai scientist agents learn from lab-in-the-loop feedback? evidence from iterative perturbation discovery. arXiv preprint arXiv:2603.26177, 2026.

[52] William Webber, Alistair Moffat, and Justin Zobel. A similarity measure for indefinite rankings. ACM Transactions on Information Systems (TOIS), 28(4):1–38, 2010.

[53] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

## 4 Methods

## 4.1 Problem formulation

We frame adaptive hit discovery as a special case of Bayesian adaptive experimental design (BAED) [37]. BAED selects experiments sequentially to maximize a utility function under uncertainty. Its components are an experimental domain $D ,$ an observation model $p ( \boldsymbol { y } \mid d , \theta )$ generating outcomes for a given experiment d and parametrized by an unknown parameter $\theta$ with prior $p ( \theta )$ , and a history $h _ { t } = \{ ( d _ { i } , y _ { i } ) \} _ { i = 1 } ^ { t }$ of past experiment-outcome pairs. At each step, the design policy $\pi ( d _ { t + 1 } \mid h _ { t } )$ selects the next experiment. For a fixed horizon T, the optimal policy maximizes

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi , p } [ U ( h _ { T } , \theta ) ] ,\tag{1}
$$

where $U ( h _ { T } , \theta )$ is a utility function over the experimental trajectory.

Let a CRISPR screen s be composed of a natural-language description $c _ { s }$ of the experimental setup, a candidate gene pool $\mathcal { G }$ of all possible genes $( | \mathcal { G } | \approx 2 0 , 0 0 0 )$ , a screen library $L \subseteq { \mathcal { G } }$ of the genes actually measured in that screen, and binary hit labels $y _ { g } \in \{ 0 , 1 \}$ for each gene $g \in L$ . At each round $t \in \{ 1 , \ldots , T \}$ , the model selects a batch $B _ { t }$ of $\mathit { \Delta } \cdot \mathit { \Delta } _ { b }$ genes, observes their hit labels, and accumulates a history $h _ { t } = \{ ( g , y _ { g } ) : g \in B _ { 1 } \cup \cdot \cdot \cdot \cup B _ { t } \}$ . The goal is to maximize the number of discovered hits within the fixed budget $T \times b \cdot$

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \Big [ \sum _ { g \in h _ { T } } y _ { g } \Big ] ,\tag{2}
$$

where the expectation is taken over full trajectories sampled from $\pi .$ This corresponds to the general BAED framework of Eq. 1 with a deterministic observation model $p ( y _ { g } \mid g ) = \mathbb { I } [ y = y _ { g } ]$ and utility $\begin{array} { r } { U ( h _ { T } ) = \sum _ { g \in h _ { T } } y _ { g } } \end{array}$ . Throughout this work we use $T = 1 0$ rounds of $b = 1 0 0$ genes, for a total budget of 1,000 acquisitions per screen.

Conventional BAED methods solve each experimental-design problem independently, fitting a taskspecific probabilistic model using only observations from the current experiment [40, 27]. Although external knowledge can be incorporated through priors, kernels, or representations, the acquisition policy itself is not learned across tasks. Amortized experimental design instead learns a shared, history-conditioned policy across a distribution of tasks, moving much of the inference cost to an offline training stage and enabling knowledge transfer to new experiments [13, 3, 23]. In our setting, completed large-scale CRISPR screens provide a collection of related experimental tasks from which sequential histories can be constructed retrospectively, allowing us to train an amortized acquisition policy directly from existing datasets without requiring a separate generative model.

## 4.2 Dataset and benchmark construction

AsSAYBENCH-LOOP builds upon AssayBench [10], a public benchmark containing 1,920 CRISPR screens. We adopt the same temporal train/validation/test split strategy and retain all 1,349 training screens. We use its temporal split: training screens before 2021, validation screens from 2021, and test screens after 2021. Because evaluating sequential design over multiple rounds is computationally intensive, we limit the validation and test sets to 20 screens each. These screens were selected to satisfy the following criteria: genome-wide coverage (18,000–22,000 genes), at least 50 hits with hit rate below 15%, a non-trivial LLM performance floor (AnDCG@ $1 0 0 \geq 0 . 0 5$ for the best LLM from AssayBench, Gemini 3.1 Pro), phenotype distribution allocated based on the distribution of eligible test set screens, with a soft cap of six screens per category, and maximized within-phenotype diversity based on TF-IDF similarity of screen descriptions. The complete dataset composition is provided in Appendix A.

## 4.3 Evaluation metrics

We let $\mathcal { G }$ denote the set of all possible genes, L the set of genes in the gene library of a given screen (for which a label $y _ { g }$ is available), H the set of hits in $L ,$ and $G _ { T }$ the set of genes acquired by the model over the T rounds. Full metric definitions and implementation details are provided in Appendix H.

Hit enrichment factor (EF). Our primary metric measures the ratio between the number of hits found and the expected number of hits under random selection. We write $N _ { L } = | G _ { T } \cap L |$ for the number of acquired in-library genes, $N _ { \bar { \mathcal { G } } } = | G _ { T } \backslash \mathcal { G } |$ for the number of hallucinated acquired genes that do not represent valid genes, and $N _ { \mathrm { m i s s } }$ for the total number of unfilled acquisition slots over the $T$ rounds. Finally, $h _ { \mathrm { r a n d } } = | H | / | L |$ denotes the random hit rate. We define

$$
\mathrm { E F } = \frac { \left| G _ { T } \cap H \right| } { \left( N _ { L } + N _ { \bar { \mathcal { G } } } + N _ { \mathrm { m i s s } } \right) \cdot h _ { \mathrm { r a n d } } } .\tag{3}
$$

This differs from the classical enrichment factor $( | G _ { T } \cap H | / ( | G _ { T } | \cdot h _ { \mathrm { r a n d } } ) )$ in the denominator: normalizing by $\left( N _ { L } + N _ { \bar { \mathcal { G } } } + N _ { \mathrm { m i s s } } \right)$ does not penalize acquisition of valid genes absent from the screen library but does penalize hallucinated genes and unfilled acquisition slots.

Additional metrics. We report the normalized area under the cumulative-hits curve (nAUC), which captures the full acquisition trajectory; the fraction of hits found (FH), measuring recall; the shortfall (SF), the fraction of acquisitions falling outside the screen library; and the percentage of acquired hits that are DepMap common-essential genes $( \% _ { \mathrm { e s s } } )$ [48], which separates screen-specific biology from generic cell-death phenotypes.

We quantify the biological diversity of the acquired genes with three measures. The effective number of Reactome pathway groups (EP) is the Hill number of order 1 of the pathway distribution induced by a set of acquired genes, reported at the batch (EP-B), screen (EP-S), and dataset (EP-D) scopes [21]. Within a batch, the Vendi score [14] is the effective number of distinct genes under a cosine-similarity kernel over GenePT embeddings [7], reported as Vendi ratio (VR), a fraction of batch size, so that higher values indicate more diverse batches. The pathway overlap (PO) is the mean pairwise Jaccard overlap of GO biological-process annotations within a batch, divided by that of a random draw from the same library: $\mathrm { P O } = 1$ matches random selection and $\mathrm { P O } > 1$ indicates genes that share annotations. Full definitions are given in Appendix H.

## 4.4 ASSAYFORMER architecture

AsSAYFORMER is a learned acquisition policy $\pi _ { \phi } ( B \mid h _ { t - 1 } , c _ { s } )$ over candidate gene batches, following the amortized design paradigm. Given a screen description $c _ { s }$ and the current history $h _ { t - 1 }$ , it produces per-gene acquisition scores over all untested candidates.

The model is a transformer encoder with $L = 3$ layers, $H = 2$ attention heads, width $d _ { \mathrm { m o d e l } } = 3 8 4$ feed-forward dimension $d _ { f f } = 1 0 2 4$ , and GELU activations. The input is a sequence of tokens comprising the experiment description (DESC) and the history gene tokens: $\big [ \mathrm { D E S C } , x _ { g _ { 1 } } , \dotsc , x _ { g _ { | h _ { t - 1 } | } } \big ]$ . No positional embeddings are applied, as gene order in the history is treated as irrelevant.

Each history gene token $x _ { g }$ combines a learnable gene embedding $v _ { g } \in \mathbb { R } ^ { d _ { g } }$ with a learned embedding $z _ { y _ { g } }$ of its binary hit label $y _ { g } \in \{ 0 , 1 \}$ . Separate embeddings $z _ { \mathrm { 0 } }$ and $z _ { 1 }$ are learned for non-hits and hits, respectively. The two are concatenated and projected to the transformer latent space:

$$
\boldsymbol { x } _ { g } = \boldsymbol { W } ^ { T } \big ( \boldsymbol { v } _ { g } \oplus \boldsymbol { z } _ { y _ { g } } \big ) ,
$$

where $W \in \mathbb { R } ^ { ( d _ { g } + d _ { h } ) \times d _ { \mathrm { m o d e l } } }$ is a learnable projection matrix.

The screen description $c _ { s }$ is processed by first embedding it with a frozen text encoder (OpenAI text-embeddi $\scriptstyle \mathtt { n g - 3 - s m a l 1 } )$ , yielding $e _ { \mathrm { d e s c } } ~ \in ~ \mathbb { R } ^ { d _ { \mathrm { t e x t } } }$ , and then linearly projecting it into the latent space: $\mathrm { D E S C } = W _ { d } e _ { \mathrm { d e s c } }$ , where $W _ { d } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { \mathrm { t e x t } } }$

The pooled output of the DESC token through the transformer is linearly projected to the gene space to produce the screen latent $\hat { u } ( h _ { t - 1 } , c _ { s } ) \in \mathbb R ^ { d _ { g } }$ . Each candidate gene $g$ is scored via a bilinear head:

$$
a _ { \phi } ( g \mid h _ { t - 1 } , c _ { s } ) \ : = \ : \hat { u } ( h _ { t - 1 } , c _ { s } ) ^ { \top } v _ { g } \ : + \ : b _ { g } ,
$$

where $b _ { g }$ is a per-gene bias. At inference time, the next batch $B _ { t }$ is constructed by taking the top-b previously untested genes according to $a _ { \phi }$

## 4.5 Training procedure

Training proceeds in three stages: gene embedding initialization, supervised fine-tuning, and reinforcement learning fine-tuning.

Gene embedding initialization. We initialize the gene embeddings $v _ { g } \in \mathbb { R } ^ { d _ { g } } ( d _ { g } = 1 0 )$ using a probit version of Bayesian Probabilistic Matrix Factorization (BPMF) [41, 1]. Let ${ Y _ { i j } } \in \{ 0 , 1 \}$ indicate whether gene $j$ scored as a hit in training screen ¿. We model

$$
Z _ { i j } = U _ { i } ^ { \top } V _ { g _ { j } } + \varepsilon _ { i j } , \quad \varepsilon _ { i j } \sim \mathcal { N } ( 0 , 1 ) , \qquad Y _ { i j } = \mathbb { I } \left[ Z _ { i j } > 0 \right] ,\tag{4}
$$

with isotropic Gaussian priors $U _ { i } \sim \mathcal { N } ( 0 , \sigma _ { u } ^ { 2 } I _ { d _ { g } } )$ on screen factors and $V _ { g _ { j } } \sim \mathcal { N } ( 0 , \sigma _ { v } ^ { 2 } I _ { d _ { g } } )$ on gene factors $( \sigma _ { u } = \sigma _ { v } = 1 )$ . Posterior distributions are estimated using Gibbs sampling with 2,000 iterations (1,000 burn-in, thinned by 2). Gene embeddings are set to the posterior mean $v _ { g } = \operatorname { \mathbb { E } } [ V _ { g } \mid Y ]$ . Genes absent from the factorization receive $v _ { q } = \mathbf { 0 }$ and a static bias at their marginal hit frequency. Algorithmic details are provided in Appendix D.1. This initialization is important for downstream performance, as discussed in Appendix B.1.

Supervised fine-tuning. The model is trained to predict hit labels for unobserved genes given simulated histories of varying length, using binary cross-entropy. For each training screen, a random subset O of the screen's genes (where $| \mathcal { O } | \sim \mathcal { U } \{ 0 , 1 0 2 4 \} ,$ ) is revealed as observed context. To ensure the model encounters informative histories, with probability 25% we resample $\mathcal { O }$ to contain at least one hit: we sample $n _ { \mathrm { h i t s } } \sim \mathcal { U } \{ 1 , \operatorname* { m i n } ( | H | - 1 , | O | ) \}$ and replace $n _ { \mathrm { { h i t s } } }$ genes in O with randomly drawn hits, capping $\mathrm { a t } | H | - 1$ to leave at least one hit unobserved. The model is trained on a balanced set O of unobserved genes (all unobserved hits plus an equal number of randomly sampled non-hits):

$$
\mathcal { L } _ { \mathrm { S F T } } ~ = ~ \frac { 1 } { | \bar { \mathcal { O } } | } \sum _ { g \in \bar { \mathcal { O } } } \mathrm { B C E } \Big ( \sigma \big ( a _ { \phi } ( g ~ | ~ h _ { t - 1 } , c _ { s } ) \big ) , ~ y _ { g } \Big ) .
$$

This yields a greedy policy $\pi _ { \mathrm { p r e } }$ that selects the b genes with highest scores at each step.

RL fine-tuning. The supervised policy is greedy and does not account for the fact that decisions at one step can influence the information gained at future steps [37]. We therefore fine-tune the policy using GRPO [42] on full T-step trajectory rollouts. For each training screen, we perform $G = 8$ rollouts. Since the policy network is deterministic, rollout diversity is produced via Gumbel-top-k sampling [26]:

$$
B _ { t } = \mathrm { T o p - b } \left( \{ \tilde { s } _ { g } : g \notin h _ { t - 1 } \} \right) ; \qquad \tilde { s } _ { g } = a _ { \phi } ( g \mid h _ { t - 1 } , c _ { s } ) + \tau \epsilon _ { g } , \quad \epsilon _ { g } \sim \mathrm { G u m b e l } ( 0 , 1 ) ,
$$

where $\tau = 1$ is the sampling temperature. Fresh noise is sampled at each acquisition step, so trajectories diverge as context accumulates.

The reward uses a “context delta" approach. At each step t, the reward is the difference in hits found by the context-conditioned policy relative to a frozen, context-free policy:

$$
r _ { t } \ = \ \big | \{ g \in B _ { t } ^ { \pi } : y _ { g } = 1 \} \big | \ - \ \big | \{ g \in B _ { t } ^ { \pi _ { 0 } } : y _ { g } = 1 \} \big | ,
$$

where $B _ { t } ^ { \pi }$ is the batch sampled by the current policy and $B _ { t } ^ { \pi _ { 0 } }$ is sampled from the same set of unobserved genes by the frozen policy $\pi _ { 0 }$ without context history. $\pi _ { 0 }$ is initialized by the supervised fine-tuning phase. Optionally, we update the weights of $\pi _ { 0 }$ from π every n epochs. This reward function is aligned with the hit-discovery objective in Eq. 2 while explicitly encouraging the model to exploit the revealed context to outperform the static prior. Advantages are computed as remaining returns $\begin{array} { r } { \bar { G _ { t } ^ { ( i ) } } = \sum _ { s > t } r _ { s } ^ { ( i ) } } \end{array}$ , normalized across the group at each step. The policy is updated by the REINFORCE gradient of Plackett-Luce logprobabilities weighted by advantages, with an entropy bonus (coefficient 0.01). The fine-tuned policy $\pi _ { \mathrm { R L } }$ is sampled greedily at inference time.

## 4.6 LLM handoff strategy

We combine the biological prior knowledge of LLMs with the adaptive capabilities of AsSAYFORMER via a handoff strategy. For the first k rounds, an LLM acts as the acquisition policy: given the screen description $c _ { s }$ and observed history $h _ { t } .$ –1 in context, it produces a list of b genes, providing a biologically informed warm start. We then switch to the amortized acquisition policy πRL for the remaining $T - k$ rounds, allowing the learned policy to adapt to screen-specific feedback accumulated during the warm-start phase. We select $k = 3$ on the validation set. We refer to the resulting composite policy as AsSAYLoOP.

## 4.7 ASSAYLLM

AsSAYLLM is an LLM acquisition policy post-trained for adaptive hit discovery. At acquisition round t, the model receives the screen description $c _ { s }$ and the observed history $h _ { t - 1 }$ as a multi-turn conversation, and returns a ranked list of b genes defining the next acquisition batch $B _ { t }$ . We train AsSAYLLM for two deployment settings: as a standalone policy controlling all T acquisition rounds, and as a warm-start policy controlling the first k rounds before handing the accumulated history to AsSAYFORMER.

Both variants use the 27B-parameter Qwen 3.6 backbone and share a two-stage post-training procedure. We first perform supervised fine-tuning (SFT) on acquisition trajectories and reasoning traces generated by GLM-5.1. This stage teaches the model to follow the sequential acquisition protocol, incorporate experimental feedback, avoid repeated selections, and reliably produce parseable gene batches. We then apply GRPO [42] to directly optimize campaign-level hit-discovery outcomes. Rollouts from the same screen form a GRPO group, controlling for variation in screen difficulty.

To define the rewards, let $V _ { t } = ( B _ { t } \cap L _ { s } ) ~ \backslash$ dom $\left( h _ { t - 1 } \right)$ denote the distinct, previously unobserved, in-library genes returned at round t, and let $| H _ { s } |$ denote the total number of hits in screen s. For the k rounds controlled by the LLM, we define

$$
R _ { \mathrm { l l m } } ( k ) = \frac { 1 } { | H _ { s } | } \sum _ { t = 1 } ^ { k } \sum _ { g \in V _ { t } } y _ { g } , \qquad C _ { \mathrm { l l m } } ( k ) = \frac { \left| \bigcup _ { t = 1 } ^ { k } V _ { t } \right| } { k b } ,
$$

$$
F _ { \mathrm { l l m } } ( k ) = \frac { 1 } { k } \sum _ { t = 1 } ^ { k } \mathbb { I } \bigg [ | V _ { t } | \geq \frac { b } { 2 } \bigg ] .\tag{5}
$$

Here, $R _ { \mathrm { { l l m } } }$ measures the fraction of screen hits found directly by AsSAYLLM, $C _ { \mathrm { l l m } }$ measures the fraction of its acquisition budget yielding distinct scoreable observations, and $F _ { \mathrm { l l m } }$ measures whether the model returns sufficiently complete and valid batches.

Standalone reward. In the standalone setting (k = T), we optimize

$$
r _ { \mathrm { s t a n d a l o n e } } = w _ { \mathrm { l l m } } R _ { \mathrm { l l m } } ( T ) - w _ { \mathrm { f m t } } \left( 1 - F _ { \mathrm { l l m } } ( T ) \right) .\tag{6}
$$

Warm-start reward. In the warm-start setting, AsSAYLLM controls rounds $1 , \ldots , k$ and the frozen AsSAYFORMER completes rounds $k + 1 , \dots , T$ . To measure handoff quality, we define

$$
R _ { \mathrm { h o } } ( k , l ) = \frac { 1 } { | H _ { s } | } \sum _ { t = k } ^ { l } \sum _ { g \in V _ { t } } y _ { g } ,\tag{7}
$$

where $V _ { k + 1 }$ is the first batch selected by AsSAYFORMER after the handoff. The warm-start reward is

$$
\begin{array} { r l } & { r _ { \mathrm { w a r m } } = w _ { \mathrm { o b j } } \mathrm { E F } + w _ { \mathrm { l l m } } R _ { \mathrm { l l m } } ( k ) + w _ { \mathrm { r e a d y } } R _ { \mathrm { h o } } ( k + 1 , T ) } \\ & { ~ + w _ { \mathrm { c o v } } C _ { \mathrm { l l m } } ( k ) - w _ { \mathrm { f m t } } ( 1 - F _ { \mathrm { l l m } } ( k ) ) . } \end{array}\tag{8}
$$

The primary term EF evaluates the complete LLM–AsSAYFORMER campaign. The $R _ { \mathrm { { l l m } } }$ term provides direct credit for hits found during the LLM-controlled rounds, while $R _ { \mathrm { h o } }$ credits the LLM for constructing a history that supports an effective first acquisition by AsSAYFORMER. The $C _ { \mathrm { l l m } }$ term encourages distinct, scoreable observations, and the format penalty discourages degenerate batches. The auxiliary terms are important because the frozen AsSAYFoRMER may recover many readily discoverable hits regardless of the warm start, making the campaign outcome alone a weak learning signal. Additional implementation details are provided in Appendix D.4.

## 4.8 Gene-gene influence analysis

To probe what AsSAYFORMER has learned about relationships between genes, we measure how observing one gene as a hit shifts the policy's predicted score for another. For an ordered probe-target pair $( p , g )$ , we draw a synthetic background history $h _ { \mathrm { b g } }$ of 50 genes sampled uniformly at random from the gene vocabulary, each assigned a uniform random binary hit label, and compare the acquisition score of the target under this history with its score under the same history extended by the probe observed as a hit. The influence of p on g is the expected difference,

$$
I ( p \to g ) \ = \ \mathbb { E } _ { h _ { \mathrm { b g } } } \Big [ a _ { \phi } \big ( g \mid h _ { \mathrm { b g } } \cup \{ ( p , 1 ) \} , c _ { s } \big ) \ - \ a _ { \phi } \big ( g \mid h _ { \mathrm { b g } } , c _ { s } \big ) \Big ] ,\tag{9}
$$

estimated by averaging over 10 independently sampled backgrounds $h _ { b g }$ with fixed random seeds. Because the per-gene bias $b _ { g }$ is shared by both terms, the difference reduces to $\big ( \hat { u } ( h _ { \mathfrak { b g } } \cup \{ ( p , 1 ) \} , c _ { s } ) - \hat { u } ( h _ { \mathfrak { b g } } , c _ { s } ) \big ) ^ { \top } v _ { g } ,$ that is, the projection of the probe-induced shift in the screen latent onto the target gene embedding. Influence is therefore directional by construction: $I ( p \to g )$ and $I ( g \to p )$ are separate quantities and need not agree in magnitude or sign.

Randomizing the hit labels of the background genes prevents the measurement from being dominated by a single hypothetical screen outcome, and the shared background across the two terms means that only the probe observation differs between them. All influences are computed with the RL-fine-tuned AsSAYFORMER checkpoint under a generic screen description (“A genome-wide CRISPR knockout screen to identify essential genes"), so that the reported relationships reflect structure learned across training screens rather than a single assay context. Probe or target genes absent from the model's gene vocabulary are excluded.

Screen-specific influences For computing screen-specific influences (Fig. 5C), given a focal screen, we replace the generic screen description with the corresponding screen description $c _ { s }$ and set the background context $h _ { b g }$ to the first 100 genes selected by Gemini-3.1-Pro (given access to $c _ { s } \ \mathrm { o n l y } )$ , together with their ground-truth hit labels.

## 4.9 Baselines

Adaptive experimental design methods. We evaluate Probability-of-Hit [40] and BioBO [27], two adaptive methods designed for CRISPR hit discovery. BioBO uses a UCB acquisition with a biological enrichment analysis prior, while Probability-of-Hit uses an active search acquisition function. Both use an MLP surrogate with biologically-informed gene embeddings but do not leverage existing CRISPR screen data.

Transfer-based baselines. To assess the value of learning from historical screens, we evaluate three baselines that exploit the training data: BPMF [41] with a greedy acquisition function (Appendix D.1); Screen-kNN (Appendix D.5), which scores candidate genes using a distance-weighted combination of hit rates from the most similar training screens; and MAML [12], which meta-learns an initialization across training screens that can be quickly adapted via gradient steps on observed hit labels. We also include a prior-hit baseline that ranks genes by their empirical hit frequency across training screens.

Large language models. We evaluate several LLM families, including GLM [53], Kimi [47], Claude [2], Qwen [36], Gemini [16], and GPT [44]. We adapt the prompt from AssayBench [10] to the sequential setting: after each acquisition round, the accumulated experimental history, including previously selected genes and their observed hit labels, is appended to the next prompt. An example is shown in Appendix E.1.

Agentic harnesses. We evaluate two agentic harnesses designed for adaptive hit discovery: LLMNN [17], which combines LLM prior knowledge with nearest-neighbor sampling, and ICBR-EF [51], which uses a hypothesis registry to improve in-context learning. Neither method leverages existing screen data. We therefore also evaluate a general coding agent on top of Haiku-4.5 with access to the training screens.

## A Dataset Composition

AsSAYBENCH-LOOP builds on AssayBench [10], a public benchmark containing 1,920 CRISPR screens. We adopt the same temporal train/validation/test split and retain all 1,349 training screens. From the validation and test splits, we select 20 screens each according to the following criteria:

1. Genome-wide coverage: the screen library contains between 18,000 and 22,000 genes.

2. Sufficient signal: the screen contains at least 50 hits, with a hit rate ≤ 15%.

3. LLM signal floor: the best LLM (Gemini-3-Pro) achieves AnDCG@ $1 0 0 \geq 0 . 0 5 .$

4. Phenotype stratification: slots are allocated proportionally to the phenotype distribution in the eligible pool, with a soft cap of six screens per phenotype.

5. Within-phenotype diversity: screens are selected via greedy max-min TF-IDF diversity over screen descriptions (phenotype, condition, cell line, cell type, and library methodology).

Supplementary Table 1 summarizes the phenotype distribution across splits. Supplementary Table 2 lists all 20 test screens with their key characteristics. Because eligibility includes a minimum LLM-signal criterion, AsSAYBENCH-LoOP is designed to evaluate adaptive discovery on screens for which the assay description contains detectable predictive signal, rather than to represent an unbiased sample of all genome-wide CRISPR screens.

Please note that different sets of genes are used in different portions of the work. There is the full training vocabulary used to fit gene representations (23,724 genes), the fixed acquisition universe G used for policy rollouts and evaluation (21,147 genes appearing in at least two screens), and each screen-specific measured library $L _ { s }$

Supplementary Table 1: Phenotype distribution across ASSAYBENCH-LOOP splits.
<table><tr><td>Phenotype category</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>Drug / Chemical / Environmental Response</td><td>278</td><td>6</td><td>6</td></tr><tr><td>Fitness / Proliferation / Viability</td><td>947</td><td>6</td><td>2</td></tr><tr><td>Host-Pathogen / Infection Response</td><td>38</td><td>6</td><td>5</td></tr><tr><td>Molecular Output / Reporter / Pathway Activity</td><td>48</td><td>2</td><td>6</td></tr><tr><td>Trafficking / Localization / Structural Phenotypes</td><td>38</td><td>0</td><td>1</td></tr><tr><td>Total</td><td>1,349</td><td>20</td><td>20</td></tr></table>

Supplementary Table 2: Test set screens in AsSAYBENCH-LOOP. Each screen is a genome-wide CRISPR screen selected from the AssayBench test split. Hit rate is the fraction of genes in the library that are hits.
<table><tr><td>ID</td><td>Phenotype</td><td>Cell line</td><td>Library</td><td>Genes</td><td></td><td>Hits Hit rate Reference</td><td></td></tr><tr><td colspan="8">Drug / Chemical / Environmental Response</td></tr><tr><td>1985</td><td>Drug response</td><td>NALM-6</td><td>CRISPRn 18,873</td><td></td><td>216</td><td></td><td>1.1% Krosl J (2022)</td></tr><tr><td>U_1736_dec</td><td>Drug response</td><td>HT-29</td><td>CRISPRn 19,008</td><td></td><td>1,189</td><td></td><td>6.3% Akinci E (2022)</td></tr><tr><td>2462</td><td>Drug response</td><td>Capan-1</td><td>CRISPRn 19,012</td><td></td><td>706</td><td></td><td>3.7%Wang LM (2023)</td></tr><tr><td>1953</td><td>Drug response</td><td>HuP-T3</td><td>CRISPRn 19,580</td><td></td><td>869</td><td></td><td>4.4%Hagel KR (2022)</td></tr><tr><td>2054</td><td>Drug response</td><td>SK-N-DZ</td><td>CRISPRa 18,652</td><td></td><td>944</td><td></td><td>5.1% Alborzinia H (2023)</td></tr><tr><td>1900</td><td>Drug response</td><td>Primary T-cells</td><td>CRISPRn 19,009</td><td></td><td>898</td><td></td><td>4.7% Carnevale J (2022)</td></tr><tr><td colspan="8">Molecular Output / Reporter / Pathway Activity</td></tr><tr><td>U_2423_merged Molecular output</td><td></td><td>Primary T-cells</td><td>CRISPRi 18,811</td><td></td><td>225</td><td></td><td>1.2% Schmidt R (2022)</td></tr><tr><td>U_1733_merged Molecular output</td><td></td><td>HeLa</td><td>CRISPRn 18,385</td><td></td><td>169</td><td></td><td>0.9% Schraivogel D (2022)</td></tr><tr><td></td><td>U_1987_merged Molecular output</td><td>HEK293T</td><td>CRISPRn 20,671</td><td></td><td>428</td><td></td><td>2.1%Wei LH (2023)</td></tr><tr><td></td><td>U_2422_merged Molecular output</td><td>Primary T-cells</td><td>CRISPRa 18,800</td><td></td><td>571</td><td></td><td>3.0% Schmidt R (2022)</td></tr><tr><td>U_2427_dec</td><td>Molecular output</td><td>CD4+ T-cells</td><td>CRISPRa 18,800</td><td></td><td>283</td><td></td><td>1.5% Schmidt R (2022)</td></tr><tr><td>2076</td><td>Molecular output</td><td>HCT 116</td><td>CRISPRn 19,007</td><td></td><td>212</td><td></td><td>1.1% Rehfeld F (2023)</td></tr><tr><td colspan="8">Host-Pathogen / Infection Response</td></tr><tr><td>1830</td><td>Infection response</td><td>HEK293T/ACE2 CRISPRn 18,984</td><td></td><td></td><td>88</td><td></td><td>0.5% Grodzki M (2022)</td></tr><tr><td>2466</td><td>Infection response</td><td>K-562</td><td>CRISPRi</td><td>18,734</td><td>88</td><td></td><td>0.5% Ngo AM (2023)</td></tr><tr><td>U_1863_inc</td><td>Infection response</td><td>Calu-3</td><td>CRISPRa 18,458</td><td></td><td>50</td><td></td><td>0.3% Rebendenne A (2022)</td></tr><tr><td>2060</td><td>Infection response Calu1-ACE2</td><td></td><td>CRISPRn 20,670</td><td></td><td>103</td><td></td><td>0.5% Ugalde AP (2022)</td></tr><tr><td>1827</td><td></td><td>Infection response HEK293T/ACE2</td><td>CRISPRn 19,007</td><td></td><td>61</td><td></td><td>0.3% Grodzki M (2022)</td></tr><tr><td colspan="8">Fitness / Proliferation / Viability</td></tr><tr><td>U_1735_dec</td><td>Fitness</td><td>HT-29</td><td>CRISPRn 19,008</td><td></td><td>126</td><td></td><td>0.7% Akinci E (2022)</td></tr><tr><td>1905</td><td>Fitness</td><td>Primary T-cells</td><td>CRISPRn 19,007</td><td></td><td>699</td><td></td><td>3.7% Carnevale J (2022)</td></tr><tr><td colspan="8">Trafficking / Localization / Structural Phenotypes</td></tr><tr><td>2404</td><td>Trafficking</td><td>HeLa</td><td>CRISPRn 19,005</td><td></td><td>89</td><td></td><td>0.5% Tsai PL (2022)</td></tr></table>

## B Ablations

We ablate three components of AsSAYFORMER: gene embedding initialization, the RL reward, and conditioning on the screen description $c _ { s }$

## B.1 Embedding initialization

We compare different gene embedding initializations: BPMF (our default, described in Section 4.5), GenePT [7], Singular Value Decomposition (SVD) of the training hit matrix, Matrix Factorization (MF) and its normalized version (MF-Sphere), and Perturb-seq derived embeddings from the K562 cell line [28]. As shown in Supplementary Table 3 (top), performance varies substantially across embedding sources. After supervised fine-tuning, EF ranges from 1.06 (GenePT) to 3.83 (BPMF). RL fine-tuning improves all initializations, but the gap between BPMF and alternatives widens: SVD is close after SFT (EF 3.72 vs. 3.83) but falls further behind after RL (4.40 vs. 4.83).

In Appendix F.2, we further observe that downstream performance does not correspond with how well the embeddings recover canonical biological relationships from STRING [46], CORUM [15], SIGNOR [30], and Reactome [34], suggesting that the structure most relevant to adaptive hit discovery diverges from standard pathway annotations. Consistent with this, the BPMF embeddings organize primarily by shared screen phenotype rather than by pathway or complex membership (Appendix F.3). The BPMF geometry is also nearly perfectly preserved during training (Pearson r = 0.999; Appendix F.1), indicating that AsSAYFORMER learns on top of this structure rather than reshaping it.

## B.2 Training strategies

Supplementary Table 3: Ablation results. Top: effect of gene embedding initialization on AsSAYFORMER performance after supervised fine-tuning (SFT) and RL fine-tuning. Bottom: training strategy ablations using BPMF embeddings.
<table><tr><td>Embedding initialization</td><td>EF (SFT)</td><td>EF (RL) (+∆)</td></tr><tr><td>Random</td><td>2.47</td><td>2.71 (+0.24)</td></tr><tr><td>GenePT</td><td>1.06</td><td>2.60 (+1.54)</td></tr><tr><td>K562</td><td>2.95</td><td>3.15 (+0.20)</td></tr><tr><td>MF</td><td>3.01</td><td>4.05 (+1.04)</td></tr><tr><td>MF-Sphere</td><td>3.72</td><td>4.31 (+0.59)</td></tr><tr><td>SVD</td><td>3.72</td><td>4.40 (+0.68)</td></tr><tr><td>BPMF (default)</td><td>3.83</td><td>4.83 (+1.00)</td></tr><tr><td>Training strategy</td><td>EF</td><td>nAUC (%) / FH (%)</td></tr><tr><td>ASSAYFORMER + GRPO</td><td>4.83</td><td>17.2 / 23.2</td></tr><tr><td>— screen description</td><td>4.52</td><td>16.2 / 22.4</td></tr><tr><td> $\mathrm { A s s a y F o R M E R + G R P O \left( E F _ { t e r m i n a l } \right) }$ </td><td>4.29</td><td>14.9 / 20.5</td></tr></table>

RL context-delta reward. During RL fine-tuning, AsSAYFORMER is trained with the context-delta reward, which explicitly rewards improvements over a context-free policy as experimental observations accumulate. We validate this choice against a natural alternative: directly optimizing the terminal EF after T = 10 acquisition rounds. As shown in Supplementary Table 3, the terminal EF objective leads to worse performance. Unlike the context-delta reward, it does not isolate the benefit of conditioning on observed history, providing a weaker learning signal for history-dependent adaptation.

Conditioning on the screen description $c _ { s } .$ Because AsSAYFORMER is primarily trained to adapt from accumulated experimental observations, we test how much it relies on the initial screen description. As shown in Supplementary Table 3, removing the screen description has relatively minor effect on performance, suggesting that initial acquisitions are driven largely by cross-screen gene hit patterns learned from the training data rather than the text of $c _ { s }$ . This limited reliance on $c _ { s }$ also helps explain the complementarity of the handoff strategy: the LLM can exploit the screen description to provide a biologically informed warm start, while AsSAYFORMER subsequently adapts from experimental feedback.

## C Complete Results

Supplementary Table 4: Full performance comparison of various baselines and ablations. Metrics evaluate enrichment factor (EF, hit rate relative to random), normalized Area Under the Curve (nAUC), fraction of hits found, mean shortfall, Vendi ratio, and batch pathway overlap versus random. EP-B/EP-S/EP-D are the effective number of Reactome level-2 pathway groups covered (186 in the vocabulary), at batch, screen and dataset scope: each gene is assigned to one of its groups at random, and the scope is subsampled to a fixed annotated-gene count (30, 200 and 6000 respectively), so the three are not directly comparable to one another.

<table><tr><td>Method</td><td>EF</td><td>nAUC (%)</td><td>Frac. hits (%)</td><td>Shortfall (%)</td><td>Ess. (%)</td><td>Vendi (%)</td><td>Path. Ov.</td><td>EP-B</td><td>EP-S</td><td>EP-D</td></tr><tr><td colspan="9">Base LLMs</td><td></td><td></td><td></td></tr><tr><td>GLM-5.1</td><td>4.00</td><td>16.3</td><td>21.0</td><td>10.6</td><td>28.9</td><td>47.2</td><td>4.72</td><td>13.3</td><td>30.8</td><td>64.0</td><td></td></tr><tr><td>- hit labels</td><td>3.65</td><td>15.0</td><td>19.1</td><td>7.3</td><td>27.6</td><td>47.0</td><td>4.44</td><td>14.4</td><td>33.4</td><td>64.6</td><td></td></tr><tr><td>GLM-5.2</td><td>4.09</td><td>16.6</td><td>21.4</td><td>5.3</td><td>29.1</td><td>47.6</td><td></td><td>4.62 14.0</td><td>34.0</td><td>64.6</td><td></td></tr><tr><td>Kimi-K2.6</td><td>3.00</td><td>14.1</td><td>15.7</td><td>54.2</td><td>31.2</td><td>60.5</td><td></td><td>6.52 13.2</td><td>29.6</td><td>64.3</td><td></td></tr><tr><td>- hit labels</td><td>2.72</td><td>13.0</td><td>14.3</td><td>37.1</td><td>31.4</td><td>56.9</td><td></td><td>4.96 14.5</td><td>32.9</td><td>62.3</td><td></td></tr><tr><td>Claude Opus-4.8</td><td>3.86</td><td>17.3</td><td>20.2</td><td>35.1</td><td>28.3</td><td>48.5</td><td></td><td>6.62 12.3</td><td>27.4</td><td>60.7</td><td></td></tr><tr><td>- hit labels</td><td>3.55</td><td>15.8</td><td>18.6</td><td>28.9</td><td>25.8</td><td>45.4</td><td></td><td>6.21 12.6</td><td>30.0</td><td>62.6</td><td></td></tr><tr><td>Claude Sonnet-4.6</td><td>3.06</td><td>13.3</td><td>16.0</td><td>29.7</td><td>29.0</td><td></td><td>51.2</td><td>5.34 14.1</td><td>34.4</td><td>65.1</td><td></td></tr><tr><td>Claude Haiku-4.5</td><td>1.39</td><td>8.5</td><td>7.3</td><td>61.8</td><td>28.9</td><td></td><td>41.9</td><td>6.75</td><td>12.5 32.7</td><td></td><td>63.5</td></tr><tr><td>Qwen3.6-27B</td><td>2.65</td><td>10.5</td><td>13.9</td><td>23.4</td><td>30.9</td><td></td><td>41.7</td><td>4.97</td><td>14.3 37.5</td><td></td><td>68.9</td></tr><tr><td>- hit labels</td><td>2.32</td><td>10.0</td><td>12.1</td><td>22.6</td><td>31.2</td><td></td><td>43.5</td><td>5.04</td><td>14.6 38.1</td><td></td><td>68.4</td></tr><tr><td>Gemini-3.1-pro</td><td>4.71 4.38</td><td>20.6 18.5</td><td>24.6 22.9</td><td>3.6</td><td>30.6</td><td></td><td>47.4</td><td>4.39</td><td>14.6 34.6</td><td></td><td>66.2</td></tr><tr><td>- hit labels</td><td></td><td>18.7</td><td>23.1</td><td>3.1</td><td>29.5</td><td></td><td>47.1</td><td>4.53</td><td>14.3 34.3</td><td></td><td>63.5</td></tr><tr><td>GPT-5.5 GPT-5.6 Sol</td><td>4.43 4.81</td><td>20.5</td><td>25.2</td><td>5.1 5.0</td><td>27.2 29.6</td><td></td><td>46.6</td><td>5.02</td><td>12.2 28.2</td><td></td><td>60.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>49.7</td><td>4.54</td><td>13.2</td><td>29.5</td><td>61.8</td></tr><tr><td colspan="10">AssayLLM (Ours)</td><td></td><td></td><td>37.5</td></tr><tr><td>Qwen3.6-27B (base)</td><td></td><td>10.5</td><td></td><td></td><td>23.4</td><td>30.9</td><td></td><td>4.97</td><td></td><td></td><td></td></tr><tr><td>+ SFT (GLM-5.1 traces)</td><td>2.65 3.56</td><td>14.7</td><td>13.9 18.7</td><td></td><td></td><td></td><td>41.7 42.2</td><td>5.36</td><td>14.3 13.6</td><td></td><td>68.9</td></tr><tr><td>+ SFT + GRPO (= AssayLLM)</td><td>3.69</td><td>15.5</td><td>19.3</td><td></td><td>19.0 16.7</td><td>35.9 37.8</td><td>41.4</td><td>5.52</td><td>13.7</td><td>34.3 34.7</td><td>64.1 64.8</td></tr><tr><td colspan="10">Adaptive Experimental Design Methods</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>1.04 2.87</td><td>3.4 11.5</td><td>4.9 14.9</td><td>10.0 1.5</td><td></td><td>18.6</td><td>58.5</td><td>1.10</td><td>21.7 56.9</td><td></td><td>82.6</td></tr><tr><td>Prior hit baseline Screen-kNN</td><td>3.40</td><td>12.5</td><td>17.3</td><td>3.7</td><td></td><td>99.1</td><td>56.5 57.6</td><td>3.22 2.91</td><td>17.8 38.4</td><td></td><td>48.4</td></tr><tr><td>RF (greedy)</td><td>1.99</td><td>6.4</td><td>9.9</td><td></td><td></td><td>77.1 30.5</td><td>50.1</td><td>3.24</td><td>18.9 18.4</td><td>42.8</td><td>54.8</td></tr><tr><td>RF +UCB</td><td>1.32</td><td>3.8</td><td>6.2</td><td>3.7 10.9</td><td></td><td>23.9</td><td>47.1</td><td>2.42</td><td>18.1</td><td>49.2 47.8</td><td>77.3 74.0</td></tr><tr><td>BPMF [41]</td><td>4.49</td><td>12.3</td><td>19.7</td><td>18.0</td><td></td><td>36.1</td><td>53.4</td><td>2.05</td><td>20.1 50.7</td><td></td><td>72.5</td></tr><tr><td>Transformer + DAgger [39]</td><td>2.80</td><td>10.3</td><td>14.4</td><td>2.5</td><td></td><td>90.0</td><td>59.8</td><td>2.52</td><td>19.0</td><td>41.9</td><td>53.9</td></tr><tr><td>MAML (w/ BPMF embs.) [12, 41]</td><td>3.77</td><td>14.3</td><td>18.6</td><td>6.4</td><td></td><td>74.9</td><td>58.2</td><td>2.54</td><td>19.4</td><td>44.4</td><td>57.8</td></tr><tr><td>BioBO [27]</td><td>2.59</td><td>5.9</td><td>12.2</td><td>10.3</td><td></td><td>38.4</td><td>64.3</td><td>1.68</td><td>19.9</td><td>50.6</td><td>72.6</td></tr><tr><td>Probability-of-hit [40]</td><td>2.41</td><td>7.3</td><td>11.8</td><td></td><td>9.0</td><td>35.2</td><td>56.6</td><td>2.11</td><td>20.3</td><td>53.4</td><td>79.0</td></tr><tr><td>Agent Harnesses</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td>Haiku-4.5 Agent</td><td></td><td>14.3</td><td></td><td>16.8</td><td>20.6</td><td>49.8</td><td>61.7</td><td>2.94</td><td>18.6</td><td></td><td></td></tr><tr><td>LLMNN [17]</td><td>2.39</td><td>9.4</td><td></td><td>12.5</td><td>0.1</td><td>22.6</td><td>64.9</td><td>1.44</td><td>20.3</td><td>47.3 52.1</td><td>68.3 78.6</td></tr><tr><td>ICBR-EF [51]</td><td>2.76</td><td>9.3</td><td></td><td>14.5</td><td>0.9</td><td>32.2</td><td>61.5</td><td>2.32</td><td>18.9</td><td>44.1</td><td>68.5</td></tr><tr></table>

## D Implementation details

## D.1 Gene embeddings initialization

Let ${ Y _ { i j } } \in \{ 0 , 1 \}$ indicate whether gene j scored as a hit in screen i, observed on the set O of screen-gene pairs actually assayed $( N = 1 , 3 4 9$ training screens, $M = 2 3 , 7 2 4$ genes (the total number in our training data), 64% of pairs observed, overall hit rate 7.3%). Each entry is generated by a latent utility

$$
Z _ { i j } = U _ { i } ^ { \top } V _ { g _ { j } } + \varepsilon _ { i j } , \quad \varepsilon _ { i j } \sim \mathcal { N } ( 0 , 1 ) , \qquad Y _ { i j } = \mathbb { I } [ Z _ { i j } > 0 ] ,\tag{10}
$$

with isotropic Gaussian priors $U _ { i } \sim \mathcal N ( 0 , \sigma _ { u } ^ { 2 } I _ { d _ { g } } )$ on the screen factors and $V _ { g _ { j } } \sim \mathcal { N } ( 0 , \sigma _ { v } ^ { 2 } I _ { d _ { g } } )$ on the gene factors $( \sigma _ { u } = \sigma _ { v } = 1 )$ . Conditioning on $Z$ renders both factor updates conjugate, so we alternate three Gibbs steps: $Z \mid Y , U , V$ drawn from unit-variance normals truncated at zero by the sign of Y [1], followed by

$$
U _ { i } \mid Z , V \sim { \mathcal { N } } \left( \Lambda _ { i } ^ { - 1 } \sum _ { j : ( i , j ) \in { \mathcal { O } } } V _ { g _ { j } } Z _ { i j } , \Lambda _ { i } ^ { - 1 } \right) , \qquad \Lambda _ { i } = \sum _ { j : ( i , j ) \in { \mathcal { O } } } V _ { g _ { j } } V _ { g _ { j } } ^ { \top } + \sigma _ { u } ^ { - 2 } I _ { d _ { g } } ,\tag{11}
$$

and the symmetric update for $V _ { g _ { j } } \mid Z , U$ . Unobserved pairs drop out of these sums, so the fit is a matrix completion rather than an imputation of zeros. We run 2,000 iterations, discard the first 1,000 as burn-in, thin by 2, and set $V _ { g _ { j } } \in \mathbb { R } ^ { d _ { g } }$ with $d _ { g } = K = 1 0$ to the posterior mean of the retained samples. Genes absent from the factorization receive $V _ { g } = \mathbf { 0 }$ and a static bias $\log \mathrm { i t } ( \hat { p } _ { g } )$ at their marginal hit frequency; genes covered by the fit carry no per-gene bias at initialization, so their score is the pure bilinear form $\hat { u } ^ { \top } V _ { g }$ and all ranking signal must come from the screen latent û that AsSAYFORMER infers in context.

## D.2 GRPO

Advantages are calculated as the remaining returns $\begin{array} { r } { G _ { t } ^ { ( i ) } = \sum _ { s > t } r _ { s } ^ { ( i ) } } \end{array}$ and normalized across the group at each step, $A _ { t } ^ { ( i ) } = ( G _ { t } ^ { ( i ) } - \mu _ { t } ) / \sigma _ { t }$ . The policy is then updated by the REINFORCE gradient of the Plackett–Luce log-probabilities weighted by $A _ { t } ^ { ( i ) }$ , with an entropy bonus (coef 0.01). Notably, the learned policy during RL acquires from and is evaluated on the full gene universe (including genes without hit labels in the given screen), so the policy learns to rank the full candidate space, not just each screen's library.

## D.3 Training settings

In the SFT phase, AsSAYFORMER was trained for 40 epochs with a batch size of 32 screens, AdamW, $\mathrm { l r } { = } 3 e - 4$ , and weight decay 1e — 2. Gene embeddings were initialized by BPMF $( d _ { g } = 1 0 )$ and left trainable (with no weight decay). The best validation epoch was selected for post-training. It was post-trained with GRPO across 4 B200 GPUs for 100 epochs with group size $8 , \gamma = 1 . 0$ , temperature 1.0. There were (floor(1349/4) = 337) steps per epoch. Rollouts were conducted using the universe of genes appearing in two or more screens (21,147 genes, same as evaluation). The context delta reward was used, with the frozen policy $\pi _ { 0 }$ updated to the current policy every 25 epochs. Training used AdamW with kl\_coef 0.0, ent\_coef 0.01, lr 1e-5, weight decay 0.0, grad-norm clipping to 1.0. This resulted in 1348 ( training set size $/ 4 ] \times 4 )$ $^ \ast 8$ (group size) \* 100 (epochs) = 1,078,400 traces during training. This is equivalent to 10,784,000 rounds of active learning and 1,078,400,000 genes acquired. We selected a final model checkpoint used in test-set evaluation was selected using validation handoff performance.

## D.4 ASSAYLLM

Backbone. Qwen3.6-27B, a dense 27B open-weight model with hybrid linear/full attention. Training uses SDPA attention throughout (FlashAttention-2 is numerically unstable for this architecture on our accelerators)

and therefore no sequence packing. All stages use bf16 parameters with DeepSpeed ZeRO-3; ZeRO-2 exceeds host memory for a model of this size.

## D.4.1 SFT

Trajectory construction. Teacher runs are T-round campaigns, one row per (run, screen, round). We reassemble them into single multi-turn conversations: a shared system prompt describing the goal, screen context, objective, hit definition, ranking criteria, and required output format; a t=1 user turn carrying the screen description $c _ { s } ;$ then, per round, an assistant turn holding reasoning traces, an answer $B _ { t }$ as a ranked comma-separated list of $b { = } 1 0 0$ genes, and a user turn holding incremental feedback. Feedback lists the labels newly added to $h _ { t } .$ , the running cumulative counts, and a non-repetition instruction. Rows whose assistant body is not a clean gene list are dropped (roughly 3%). Only assistant tokens are unmasked in the loss.

Optimization. Full-parameter fine-tuning across 16 GPUs with ZeRO-3, gradient checkpointing, and SDPA attention; 3 epochs, global batch size 32, learning rate $1 \times 1 0 ^ { - 6 }$ with 3% warmup.

## D.4.2 GRPO

Rollout. One prompt is one screen. The policy generates $B _ { 1 } , \ldots , B _ { k }$ with reasoning enabled; after each round the environment parses the gene list, reveals $y _ { g }$ for picks in $\mathcal { G } _ { s }$ , and appends the incremental feedback turn. Group size is 16 samples per prompt, sampled at temperature 1.0 and top-p 0.95; higher diversity is useful here because group-relative advantages need within-group spread.

Optimization. We then optimize the policy with Group Relative Policy Optimization (GRPO). For each prompt x we sample a group of G responses $\{ y _ { i } \} _ { i = 1 } ^ { G }$ from the current policy and compute the groupnormalized advantage

$$
\hat { A } _ { i } = \frac { r _ { i } - \mathrm { m e a n } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } ,\tag{12}
$$

and maximize the clipped surrogate objective

$$
\mathcal { I } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } \Big ( \rho _ { i } \hat { A } _ { i } , ~ \mathrm { c l i p } ( \rho _ { i } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i } \Big ) \right] - \beta \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) ,\tag{13}
$$

where $\rho _ { i } = \pi _ { \theta } ( y _ { i } \mid x ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i } \mid x )$ . Group-normalized advantages, global batch size 48, actor learning rate $1 \times 1 0 ^ { - 6 }$ with 3% warmup, ZeRO-3 / bf1 6, non-reentrant gradient checkpointing, SDPA attention.

## D.5 Screen-kNN implementation

Screen-kNN predicts gene hits by finding the most similar historical screens in the training set and aggregating their outcomes. Let $\boldsymbol { \mathcal { S } }$ denote the set of training screens. Following the notation of the main text, each training screen $s \in { \mathcal { S } }$ has a gene library $L _ { s } \subseteq { \mathcal { G } }$ and binary hit labels $y _ { g } ^ { s } \in \{ 0 , 1 \}$ for $g \in L _ { s }$ i

Prior. We compute a global prior for each gene as its empirical hit rate across all training screens in which it appears:

$$
p _ { 0 } ( g ) = \frac { \sum _ { s \in S } y _ { g } ^ { s } \mathbb { I } [ g \in L _ { s } ] } { \sum _ { s \in S } \mathbb { I } [ g \in L _ { s } ] } .\tag{14}
$$

Screen similarity. At round t, the policy conditions on the history $h _ { t - 1 }$ , whose gene set $G _ { t - 1 }$ carries the labels $\{ y _ { g } \} _ { g \in G _ { t - 1 } }$ observed so far in the current screen. For each training screen s, we compute the normalized Hamming distance over the genes shared with that history:

$$
\delta ( s ) = \frac { 1 } { | G _ { t - 1 } \cap L _ { s } | } \sum _ { g \in G _ { t - 1 } \cap L _ { s } } ( y _ { g } - y _ { g } ^ { s } ) ^ { 2 } .\tag{15}
$$

If $G _ { t - 1 } \cap L _ { s } = \emptyset$ , we set $\delta ( s ) = 1$ . Distances are converted to weights via an exponential kernel with temperature τ:

$$
w _ { s } = \exp \left( - \frac { \delta ( s ) } { \tau } \right) .\tag{16}
$$

Prediction. The screen evidence for gene $g$ is the weighted hit rate across training screens whose library contains $g \colon$

$$
p _ { \mathrm { s c r e e n } } ( g ) = \frac { \sum _ { s : g \in L _ { s } } w _ { s } y _ { g } ^ { s } } { \sum _ { s : g \in L _ { s } } w _ { s } } .\tag{17}
$$

The acquisition score is a Bayesian-style mixture of the prior and the screen evidence,

$$
a _ { \mathrm { K N N } } ( g \mid h _ { t - 1 } ) = ( 1 - \lambda _ { t } ) p _ { 0 } ( g ) + \lambda _ { t } p _ { \mathrm { s c r e e n } } ( g ) ,\tag{18}
$$

where $\lambda _ { t } = | G _ { t - 1 } | / ( | G _ { t - 1 } | + \kappa )$ interpolates between the prior (when few genes have been revealed) and the screen evidence (as more observations accumulate). We use $\tau = 0 . 1$ and $\kappa = 1 0 0$ , and form the batch $B _ { t }$ from the b untested genes with the highest score.

## E Example Inputs

## E.1 LLM Prompt Examples

The following is an example active-learning prompt AFTER round 1 for the following screen:

• Screen: 2054

• Cell line: SK-N-DZ (Neuroblastoma)

• CRISPRa

• RSL3 ferroptosis-resistance screen

• TEST set.

## System Prompt

You are a computational biologist. Given a CRISPR screen description, produce a ranked list   
of 100 gene symbols (HGNC nomenclature for Homo sapiens) that you believe are MOST LIKELY   
to be hits in this screen, ordered strongest first. Use only your knowledge of biology --no   
chain-of-thought in the output. Return ONLY a comma-separated list of 100 symbols, no   
prose, no numbering.

User Prompt   
## Goal   
You are tasked with ranking genes from a genetic perturbation screen. Based on the   
experimental context and hit criteria provided below, provide a list of exactly 100 genes   
that are hits in this screen, ranked from strongest to weakest according to the criteria   
defined below.   
## Experimental Context   
This screen was performed in SK-N-DZ cells, a Neuroblastoma Cell Line. Researchers used a   
CRISPRa library (Activation) to systematically perturb gene function. The experiment   
followed a Drug Exposure design and was conducted over 14 Days under 1S,3R-RSL3 treatment   
(300.0 nM).   
## Screen Objective   
The primary objective of this screen was to identify a set of hit genes, each of which   
increases resistance to the ferroptosis-inducing drug RSL3, as measured by increased cell   
viability after exposure.   
## Hit Definition   
A gene is classified as a "hit" if its Activation significantly increases resistance to the   
ferroptosis-inducing drug RSL3, as measured by increased cell viability after exposure.   
The statistical criterion for significance is: FDR < 0.05.   
## Ranking Criteria   
Genes with low FDR are ranked most highly.   
## Additional Context   
Screen notes: A genome-wide CRISPR-activation (CRISPRa) screen was performed to identify   
novel regulators of ferroptosis. This CRISPRa\_RSL3 screen involved positive selection for   
viability after exposing the cells to the ferroptosis inducing agent RSL-3 which inhibits   
GPX4   
## Required Output Format   
Provide your response as an ordered list of exactly 100 HGNC gene symbols, using the   
ranking criteria above.   
That is, top genes should have low FDR.   
Format:   
GENE1, GENE2, GENE3, ..., GENE100   
## Active-Learning History   
You have already validated 99 gene(s) in this screen, shown below grouped by round in   
acquisition order. Do NOT re-suggest any of them.   
### Round 1 (99 genes, hit rate 12/99 = 12.1%   
Hits (12): GPX4, AIFM2, GCH1, MBOAT2, GCLM, LRP8, SLC7A11, DGAT1, CHMP4A, NQO1, ME1, MUC1   
Non-hits (87): DHODH, MBOAT1, SCD, ACSL3, FTH1, FTL, NFE2L2, AKR1C1, AKR1C2, AKR1C3,   
PROM2, SLC40A1, COQ2, COQ6, COQ8A, COQ9, PDSS1, PDSS2, PTS, SPR, DHFR, QDPR, GSS, GCLC,   
TXNRD1, TXN, SQSTM1, HMOX1, MT1G, HSPA5, HSPB1, ALDH3A1, SECISBP2, SEPSECS, EEFSEC,   
SEPHS2, SLC3A2, CAV1, ABCB1, ABCG2, ABCC1, PLIN2, PLIN3, PLIN4, DGAT2, SREBF1, FASN,   
CHMP5, CHMP6, VPS4A, VPS4B, CHMP1A, CHMP1B, CHMP2A, CHMP2B, CHMP3, CHMP4B, CHMP7, IST1,   
PRDX1, PRDX6, SRXN1, GSR, IDH1, PGD, G6PD, TKT, TALDO1, ATF4, ASNS, PCK2, OTUB1, CD44,   
SOX9, HIF1A, EIF2S1, EIF2AK3, EIF2AK4, PCBP1, PCBP2, COQ3, COQ4, COQ5, COQ7, MVD, FDPS,   
HMGCR   
Cumulative: 12/99 hits (12.1%   
=== Final instruction ===   
Return ONLY a comma-separated list of 100 gene symbols (no prose, no numbers, no markdown).   
All gene symbols must use the HGNC convention for Homo sapiens (e.g. 'TP53' for HGNC,'   
Trp53' for MGI). Do not return symbols from another organism.

![](images/07156e86394792a6f1de3536fabfeb00e4217e717d1df761f3ec1f9ad50766b8.jpg)

## E.2 Qualitative analysis of ASSAYLOOP on two biologically distinct screens

![](images/bc78d16e866f1f1e819416bf98c1e90986c8aa6d879f26c47db04633ac4f40fb.jpg)  
Supplementary Figure 1: Qualitative Analysis of AsSAYLOOP on two biologically distinct screens. We show the composition of genes of suggested acquisition rounds across three models: AsSAYFORMER (top row), Gemini-3.1-Pro (bottom row), and the AsSAYLOOP handoff from Gemini (middle row). On the left side of the figure, we have a NF-κB / TNF signaling screen. On the right side, we have a AAV transgene silencing screen. For each subplot (representing a screen-model pair), we have two histograms. The upper histogram shows the proposed acquisition batch at each acquisition round. The lower histogram shows the composition of hits from that batch. For example, in the AAV screen, AsSAYFORMER proposes 100 genes for the first batch, with DNA repair being the largest category. The bars are colored according to gene pathways from Reactome.

## F Gene Embedding Analysis

This appendix provides additional analysis of the gene embeddings used by AsSAYFORMER, complementing the ablation in Section B.1.

## F.1 Embedding drift during training

A natural question is whether AsSAYFORMER reshapes the gene embedding geometry during training or preserves the structure imposed by the initialization. We examine this for all ablated embedding types by measuring two complementary quantities: pairwise cosine similarity preservation and gene-neighborhood stability.

Pairwise cosine similarity. For each embedding type, we compute the pairwise cosine similarity between 300,000 randomly sampled gene pairs before and after training. As reported in the main text (Supplementary Fig. 2), BPMF embeddings are almost perfectly preserved (Pearson $r = 0 . 9 9 9 )$ . This indicates that the geometry provided by BPMF is already well-suited for the acquisition task and that AsSAYFORMER learns its policy on top of this fixed structure rather than reorganizing it.

Gene-neighborhood stability. We further quantify structural changes using rank-biased overlap (RBO) [52] between the 100 nearest neighbors of each gene before and after each training stage. For BPMF, the median RBO between initialization and the final RL model exceeds 0.95, confirming that gene neighborhoods are largely unchanged.

Supplementary Fig. 2 shows the degree of embedding drift across all ablated embedding types. The drift varies dramatically across sources. Random embeddings undergo near-complete reorganization during supervised fine-tuning, as expected. Other matrix-factorization variants (SVD, MF, MF-Sphere) also show substantial neighborhood changes during supervised fine-tuning, though less than random initialization. Interestingly, GenePT embeddings show relatively little drift despite their poor downstream performance, suggesting that embedding stability alone does not explain the advantage of BPMF. In all cases, the RL fine-tuning stage produces much smaller changes than supervised fine-tuning.

Together, these results suggest that the effectiveness of BPMF stems not merely from stability but from providing a geometry that is both stable and well-aligned with the adaptive hit-discovery objective. Other embeddings are either reshaped during training (random, SVD, MF) without converging to an equally effective geometry, or are stable but encode structure less relevant to the task (GenePT).

## F.2 Embedding initialization and biological plausibility

As discussed in Section B.1, we observe a negative association between how well an embedding source recovers canonical biological relationships and its downstream performance as an AsSAYFORMER initialization. Supplementary Fig. 3 visualizes this relationship. For each embedding type, we compute pairwise cosine similarity between genes and measure the AUROC for recovering known gene-gene relationships from STRING [46], CORUM [15], SIGNOR [30], and Reactome [34], averaged across the four databases. GenePT and K562 embeddings, which capture these known relationships best, provide the weakest initialization for AsSAYFORMER. Conversely, matrix-factorization approaches that learn directly from historical screen data perform best, despite encoding less canonical biology. This suggests that the gene-gene structure most relevant to adaptive hit discovery is not well captured by established pathway and interaction databases.

![](images/71c4ecfd2dff5b0c7c3a7a4716dae5a7b8a3aabbe4490721ffe82010b778b270.jpg)  
Supplementary Figure 2: Embedding drift during training for all ablated embedding types. The effect of training on embedding geometry differs dramatically by embedding type. BPMF embeddings show near-perfect preservation of gene neighborhoods, while other initializations undergo substantial reorganization during supervised fine-tuning.

![](images/50af79afffaac21d6bd63b0ec9c900baf47b5c0958f50b8027454161bda69734.jpg)  
Supplementary Figure 3: Recovering textbook biology does not predict task usefulness. We take initial embeddings and measure how much they recover gene-gene relationships from four well known databases: STRING, CORUM, SIGNOR, and REACTOME. We compute gene-gene similarity scores with cosine similarity. Ground truth labels are assigned to be 1 if the relationship exists in the database, otherwise 0. AUROC is calculated using these scores and then averaged across the four databases. Interestingly, we find that embeddings which capture these databases most closely (e.g., GenePT, K562 embeddings) are more difficult to train AsSAYFORMER on, whereas methods which initialize directly from historical screen data (matrix factorization approaches) are most useful.

![](images/dbf7a0ecd74a37fbfc77d5b446c0e5c2a56a1f8c4583c685909101cf5e8b25d5.jpg)  
Supplementary Figure 4: BPMF embeddings organize by screen phenotype, rather than physical complex or pathway. Embeddings are K = 10 dimensions and are visualized with PCA (left) and cosine UMAP (right). Top: HDBSCAN. Genes are clustered in the original space with HDBSCAN [32] with minimum cluster size 150, each labeled by its dominant phenotype association. Note that several clusters are associated with DNA-damage response without further disambiguation. Middle: CORUM Complexes. The embeddings are colored by CORUM complexes grouped into general families (top 10 groups, grey = other). This coloring shows minimal grouping across the space. Bottom: Reactome pathways. The largest 10 Reactome pathways are used to color the embeddings, similarly revealing a lack of distinct spatial clustering except in the common essential gene area.

## F.3 Structure of the BPMF embedding space

To better understand why BPMF embeddings are effective, we visualize and cluster the $d _ { g } = \mathrm { 1 0 - d i m e n s i o n a l }$ embedding space. Supplementary Fig. 4 shows PCA and cosine-UMAP projections of the BPMF gene embeddings, colored by three different annotation schemes.

HDBSCAN clustering. We cluster genes in the original embedding space using HDBSCAN [32] with a minimum cluster size of 150. The resulting clusters align primarily with screen phenotype categories rather than molecular pathway or complex membership. We identify clusters corresponding to common essential, context-essential, and selectively essential genes, as well as clusters associated with immune response and apoptosis. Notably, five distinct clusters are associated with the DNA-damage response, which we were unable to further disambiguate into more specific phenotypes.

Canonical annotations. We also color the embeddings by CORUM protein complexes and Reactome pathways, each grouped into high-level families. In both cases, the coloring reveals minimal spatial clustering: genes belonging to the same complex or pathway are distributed across the embedding space rather than forming compact groups. The exception is the common-essential gene region, which shows partial organization by Reactome pathway membership.

Interpretation. These observations indicate that the BPMF embedding space is organized primarily by shared screen phenotype — reflecting how genes co-occur as hits across historical CRISPR screens — rather than by canonical pathway or complex membership. This is consistent with the finding in Section B.1 that embeddings which better recover canonical biological relationships (STRING, CORUM, SIGNOR, Reactome) provide weaker initialization for AsSAYFORMER. The BPMF embeddings capture a distinct, phenotype-derived representation of gene function, in which proximity reflects shared perturbation profiles across screens rather than established biological annotations. Their strong performance suggests that these phenotype-derived embeddings may complement existing gene embedding approaches [28] and motivates their evaluation in other downstream tasks.

## G Biological Diversity Analysis

Supplementary Fig. 5 breaks down the Reactome composition of acquired genes by individual LLM, complementing the method-level comparison in Figure 5A.

![](images/5d5faaa965a5683d9091f1c6c3605ce2deac1ce5b6a3b15c4df845a2b21355c0.jpg)

Supplementary Figure 5: LLMs converge to similar distributions of biology across families. The heatmap shows top-level composition of requested genes in terms of Reactome pathways, aggregated across the 20 test screens. Unique genes is the total number of distinct genes requested across all tested screens. Annotated picks is the percent of requested genes which have ground truth labels. The Random baseline is the pathway distribution given by uniform sampling over the universe of all possible genes. Cells are colored by by the ratio between the observed pathway share for that model and the expected share size of Random, where red = over-represented and blue = under-represented (color values capped between 0.5x and 2x). Overall, LLMs generally show similar gene acquisition signatures, with an emphasis on RNA and protein metabolism and under-representation of bulk metabolism and developmental biology. While there are differences between models, they are small compared to the difference from other baseline methodologies such as kNN. Additionally, the number of effective pathways for each LLM falls into a narrow range. Note that Kimi-K2.6 and Opus tend to supply less than the requested number of genes (100) in many cases, possibly due to biological safety training, resulting in a low number of unique genes.

## H Metrics used in ASSAYBENCH-LOOP

Below we define the metrics used in AsSAYBENCH-LOOP to evaluate the performance of hit discovery models. We let $\mathcal { G }$ denote the set of all possible genes, L the set of genes in the gene library of a given screen (the set of genes for which a label $y _ { g }$ is available), H the set of hits in L, and $G _ { T }$ the set of genes acquired by the model over the T rounds. Supplementary Fig. 6 provides a visual overview of the metrics and their definitions.

Hit enrichment factor (EF) Our main metric is the hit enrichment factor (EF), that measures the ratio between the number of hits found and the expected number of hits found by random selection.

![](images/7fea05179a8344c82ff01a1dc243c0529cd6dcc4c8d475b911628b79e338ed08.jpg)  
Supplementary Figure 6: Example of Enrichment Factor and other metrics. A) The definition of our adjusted enrichment factor (EF). B) Terms used for normalizing budget calculations. C) A worked example of EF showing how EF is comparable across screens. Note, no adjustment for untested genes is needed in this computation. D) Example of hit recovery trajectories from real baselines. E) The definition used to calculate adjusted normalized AUC (nAUC). F) Additional metrics used for understanding model behavior.

We write $N _ { L } = | G _ { T } \cap L |$ the number of acquired genes over the full trajectory present in the screen library, $N _ { \mathcal { G } } = | G _ { T } \cap \mathcal { G } |$ the number of valid acquired genes, $N _ { \bar { \mathcal { G } } } = | G _ { T } \backslash \mathcal { G } |$ the number of hallucinated acquired genes that do not represent valid genes, and $N _ { \mathrm { m i s s } }$ for the total number of unfilled acquisition slots over the $T$ rounds. Finally, $h _ { \mathrm { r a n d } } = | H | / | L |$ denotes the random hit rate. We define the hit enrichment factor as

$$
\mathrm { E F } = \frac { \left| G _ { T } \cap H \right| } { \left( N _ { L } + N _ { \bar { \mathcal { G } } } + N _ { \mathrm { m i s s } } \right) \cdot h _ { \mathrm { r a n d } } } .\tag{19}
$$

This formula differs from the classical enrichment factor $( i . e . , \frac { | G _ { T } \cap H | } { | G _ { T } | \cdot h _ { \mathrm { r a n d } } } )$ in the number of acquired genes that are used for the normalization. By normalizing with $( N _ { L } + N _ { \bar { \mathcal { G } } } + N _ { \mathrm { m i s s } } )$ , we do not penalize acquisition of real genes $( g \in { \mathcal { G } } )$ that are not part of the screen library but do penalize the acquisition of hallucinated genes.

Normalized area under the cumulative-hits curve (nAUC) We also report the normalized area under the cumulative-hits curve (nAUC), which captures the full acquisition trajectory rather than only the endpoint We construct the curve by plotting the fraction of hits found against the fraction of the library effectively queried, with one point per acquisition batch. For each batch $t = 1 , \dots , T$ , we denote $N _ { L , t }$ and $N _ { \bar { \mathcal { G } } , t }$ for the cumulative number of in-library and hallucinated acquired genes in $G _ { t }$ respectively, and $N _ { H , t } = | G _ { t } \cap H |$ for the number of hits in $G _ { t }$ . Unlike in Eq.19, unfilled acquisition slots do not advance $x _ { t } ,$ so an under-supplying policy ends its curve at a smaller x rather than being carried flat to the full budget. This keeps nAUC a measure of ordering quality. The cumulative-hits curve is then composed of points with coordinates $\left( \frac { 1 } { | L | } ( N _ { L , t } + N _ { \bar { \mathcal { G } } , t } ) , \frac { 1 } { | H | } N _ { H , t } \right)$ for each batch.

'The area under this piecewise-linear curve is computed by trapezoidal integration and normalized by the AUC of a perfect oracle that ranks all hits before any non-hit, over the same effective budget. The perfect oracle AUC is computed over a curve with coordinates $\left( x _ { t } , m i n ( x _ { t } \cdot | L | / | H | , 1 ) \right)$ , where $x _ { t }$ are the coordinates of the predictor curve $\begin{array} { r } { ( x _ { t } = \frac { 1 } { | L | } ( N _ { L , t } + N _ { \bar { \mathcal { G } } , t } ) ) } \end{array}$

$$
\mathrm { n A U C } = \frac { \mathrm { A U C } } { \mathrm { A U C } _ { \mathrm { o r a c l e } } } ,\tag{20}
$$

where $\mathrm { { A U C } _ { \mathrm { { o r a c l e } } } }$ is the area under the optimal curve evaluated at $x = ( N _ { L } + N _ { \bar { q } } ) / { | L | }$ . A value of 1.0 indicates the model acquires all hits in the screen first. Random acquisition yields $\mathrm { n \bar { A } U C } \approx | H | / | L |$ (the hit rate) in expectation when $\left| G _ { T } \right| < \left| H \right|$

Fraction of hits (FH) We report the fraction of hits found over the whole trajectory, measuring the recall of the acquisition policy. Using the same notation, this is defined as

$$
\mathrm { F H } = \frac { | G _ { T } \cap H | } { | H | } ,\tag{21}
$$

i.e. the number of hits acquired by the model divided by the total number of hits in the screen. Only in-library genes can be hits, so out-of-library and hallucinated acquisitions do not contribute to the numerator. A value of 1.0 means all hits in the screen were found within the acquisition budget.

Shortfall (SF) We report the shortfall as the fraction of acquired genes that fall outside the screen's gene library. Using the same notation:

$$
\mathrm { S F } = \frac { | G _ { T } \setminus L | } { | G _ { T } | } ,\tag{22}
$$

i.e. the number of acquired genes not present in the screen library divided by the total number of acquisitions. This includes both real genes outside the library $( G _ { T } \cap ( \mathcal { G } \setminus L ) )$ and hallucinated genes $( G _ { T } \backslash { \mathcal { G } } )$

Percentage of essential genes $( \% _ { \mathbf { e s s } } )$ Some genes are hits in a given screen just because these genes typically lead to cell death in general, also known as essential genes. While these genes are genuine hits, they do not always reflect the particularities of the biology studied in the screen. We report their fraction among recovered hits as a diagnostic of how strongly acquisition is concentrated on broadly essential genes. We used the list of 1,827 common essential genes from the Cancer Dependency Map (DepMap) [48].

Let E denote the set of common essential genes, the percentage of essential genes is simply

$$
\% _ { \mathrm { e s s } } = \frac { | G _ { T } \cap H \cap E | } { | G _ { T } \cap H | } ,\tag{23}
$$

i.e. the number of acquired hits that are common essentials divided by the total number of acquired hits.

Effective Pathways (EP) We quantify the biological breadth of acquired genes using the effective number of Reactome level-2 pathway groups (186 groups in total). This is calculated using the first order Hill number

$$
E P = \exp ( H ( p ) )
$$

where H is Shannon entropy and $p$ is the distribution of pathways in the sample set (assuming they were sampled uniformly).

To make methods with different acquisition and annotation rates comparable, we rarefy each scope to a fixed number of annotated genes: $M = 3 0$ per batch (EP-B), M = 200 per screen (EP-S), and $M = 6 0 0 0$ across the complete test set (EP-D). Sample sets containing fewer than the required number of annotated genes are excluded. We average over 400 Monte Carlo draws for EP-B and EP-S and 300 draws for EP-D. EP-B is averaged within each screen and then across screens, whereas EP-S is averaged directly across screens. Because a gene may be annotated to multiple groups, in each Monte Carlo draw we assign every annotated gene to exactly one of its associated groups. Assigning one group per gene prevents heavily annotated genes from artificially increasing diversity.

Thus, EP is expressed as an effective number of equally represented pathway groups, with larger values indicating broader biological coverage.

Batch diversity metrics: Vendi score (VS) and Pathway overlap (PO) We report two complementary diversity metrics for the acquired batches, measuring whether the model selects functionally diverse genes or concentrates on a narrow biological neighborhood at each step.

Vendi Ratio (VR). For each batch, we compute gene embeddings using GenePT [7], construct the cosine-similarity Gram matrix K of all acquired genes in the batch $B _ { t }$ , and compute its Vendi score [14], which corresponds to the effective number of distinct genes in $B _ { t }$ (1 means all embeddings of genes in $B _ { t }$ are collinear and $| B _ { t } |$ means all embeddings are orthogonal). We report the Vendi ratio, which is normalized by $| B _ { t } |$ and averaged across batches. A higher Vendi ratio means the genes acquired in the batch are more diverse.

Pathway overlap (PO). For each batch, we compute the mean pairwise Jaccard overlap of GO biologicalprocess annotations between the acquired genes, and divide it by the same quantity for a batch of equal size drawn uniformly at random from the screen library. Averaging across batches gives PO. A value of 1.00 corresponds to random selection, values above 1 indicate that acquired genes share more annotations than expected by chance (a narrow biological neighborhood), and values below 1 indicate a broader batch. PO and VS are complementary: VR measures diversity in a continuous embedding space, whereas PO measures it through discrete pathway membership.

## I ASSAYFORMER Reveals Directional Gene-Gene Relationships

![](images/5d490898318ca14c4378758b0c44c2c1b7308eb91dd1e1836fce1e5a712accc9.jpg)  
Supplementary Figure 7: Influence heatmap showing how discovered hits affect predictions. Given a probe gene (rows) and target gene (columns), we measure the change in the acquisition score of the target gene from AsSAYFORMER when the probe gene is reported as a hit in context. The first 12 genes are important canonical cancer drivers (oncogenes and tumor suppressors). The next 19 are representative functional-module genes.

Because AsSAYFoRMER updates its predictions as experimental observations accumulate, we can probe the learned policy to ask how observing one gene as a hit changes its predictions for other genes. For a probe-target pair, we construct a random background context of 50 genes, measure the acquisition score of the target, and then measure it again after adding the probe gene to the context as a hit. We define the difference between these predictions as the influence of the probe on the target, and average this quantity across 10 independently sampled background contexts. So we can inspect model-wide influence patterns learned by the model, we use a generic screen description “A genome-wide CRISPR knockout screen to identify essential genes".

Supplementary Fig. 7 shows the resulting influence matrix for 31 representative genes, including 12 canonical cancer drivers and 19 genes from functional modules. Influences among cancer drivers are relatively weak, consistent with these genes acting through distinct biological programs. In contrast, substantially stronger influences emerge among functionally related gene pairs, indicating that AsSAYFORMER has learned structured, context-dependent relationships from historical screens.

![](images/3f982cb91c980f1b742024072aa9c421a6a431e78196c2b9c3c8e4154fabb019.jpg)

![](images/6810a576a134494c0787536dfdfc21f0ac82690312a1136539565e756f2fe1ca.jpg)  
Supplementary Figure 8: Gene influence scores reveal hidden epistatic relationships not captured by standard interaction databases. For each probe gene (left of arrow), we measure the change in the model's predicted influence score for every target gene (right of arrow) when the probe gene is observed as a hit, averaged over random background contexts. Left: Boosted pairs. Right: Suppressed pairs. These dependencies describe the model's inferred decision process rather than established biological interactions.

Importantly, these relationships are strongly directional rather than simple symmetric associations. The correlation between the influence matrix and its transpose is only r = 0.16, and 44% of reciprocal gene pairs have influences with opposite signs. For example, observing MDM2 as a hit increases the acquisition score of PFDN4 by +0.33, whereas observing PFDN4 as a hit decreases the acquisition score of MDM2 by —0.46. Thus, the model does not simply encode gene similarity: observing a gene can induce a directional update in the predicted relevance of another gene, and the effect need not be reciprocal.

To test whether this procedure can uncover meaningful relationships beyond canonical annotations, we remove gene pairs represented in the STRING, CORUM, SIGNOR, MSigDB, and Reactome databases [46, 15, 30, 45, 34] and examine high-magnitude remaining influences, shown in Supplementary Fig. 8. Among positively influenced pairs, we identify relationships consistent with co-essentiality or synthetic-lethal dependencies, such as increased spliceosome dependence following MYC hits and increased nucleolarstress dependence following MDM2 hits. Conversely, negative influences highlight patterns consistent with epistatic masking or lineage-specific exclusion, including reduced dependence on mitotic machinery following PIK3CA hits and reduced OXPHOS dependence following SMAD4 hits.

## J Model Scaling

We investigate whether performance gains similar to those from additional training data can be obtained by increasing model capacity. We train 150 models spanning a nearly 40-fold range in parameter count, from 0.28M (XS) to 10.97M (XL), including our default 4.44M-parameter model (L). In contrast to data scaling, we observe no meaningful improvement with increasing model size (Supplementary Fig. 9). This suggests that, in the current data regime, available training screens rather than model capacity are the binding constraint on performance.

![](images/1eaf5385cffa1f50b1e3bb40abe3a665f99d95675972f5fb3a61bb79a40d570c.jpg)

Supplementary Figure 9: Performance scales with data, but not model size. ASSAYFORMER models (without input screen descriptions) are trained to identify scaling of in-context learning capabilities. Top: Models are trained on varying amounts of data from 1 to all training screens, and average performance curves are plotted. Both SFT and RL show this performance increase, with RL boosting SFT performance. Top Left plot is EF at k acquisition steps, and shows that scaling is most effective during the first few steps of the experiment. At 100 steps (10,000) genes sampled, models approach similar performance. This is expected because even a random model will find all hits if it samples the entire genome. Top Right plot is budget to acquire enough hits to satisfy a recall threshold m. For example, to find m = 50% of the hits, the 1349' model requires sampling less than 30% of the full genome. Note that RL rollouts stop at 10 steps during training, so these curves reflect out-of-distribution gains for RL performance. Bottom: We also consider scaling model size from XS (0.28M parameters) to XL (10.97M parameters), given the full training set Here, we do not observe meaningful scaling laws. We posit that larger training datasets are required to show evidence of model scaling. Note that scaling models has shown useful performance gains in pretrained LLMs on the non-active learning setting of AssayBench (Figure 4 from [10]).

## K Revisiting whether LLMs can learn from lab-in-the-loop feedback

Recent work has investigated whether LLMs can serve directly as acquisition policies in lab-in-the-loop biological experiments. Gupta et al. [17] found that the performance of several LLM-based experimentaldesign agents was largely insensitive to feedback provided during the experiment, whereas Wainrib et al. [51] showed that sufficiently capable LLMs can benefit substantially from experimental feedback. We revisit this question on AsSAYBENCH-LOOP by evaluating a range of LLMs with and without access to the outcome labels from previous acquisition rounds. Unlike these prior studies, which randomized hit assignments in their control conditions, we simply omit the hit labels while preserving the history of previously acquired genes. This provides a direct measure of how much each model benefits from observing experimental outcomes. We additionally compare to AsSAYFORMER without context; in this setting, we simply take the top 1,000 genes predicted by AsSAYFORMER with no context.

As shown in Supplementary Figure 10, removing hit labels consistently reduces performance across all tested LLMs, demonstrating that these models do use experimental feedback to adapt their subsequent acquisitions. This result is consistent with Wainrib et al. [51] and confirms that adaptive in-context learning extends across multiple LLM families. However, the magnitude of this adaptation remains limited: standalone LLMs are consistently outperformed by AsSAYFORMER, which shows a substantially larger gain from feedback. Furthermore, AsSAYLoOP, which utilizes an LLM for early acquisitions and delegates later acquisition rounds to AsSAYFORMER, shows both strong label-blind performance and also shows better in-context learning than LLMs. Thus, while current frontier LLMs can learn from experimental outcomes in context, a policy explicitly trained for feedback-conditioned adaptation remains more effective for sequential hit discovery.

![](images/14c4330e687fe5f86507a61af77dda7ff1f8c3586bd077bcf13af499b8dd1faf.jpg)  
Supplementary Figure 10: LLM performance with and without per-round hit labels. We tested the in-context learning capabilities of select LLMs by running the active learning framework without labels from previous rounds of experiments. This allows us to distinguish between the ability of LLMs to incorporate experimental feedback with ICL, versus their warm-start capabilities without feedback. Across model families, we found that experimental feedback boosts model performance, indicating that LLMs do benefit from the active learning setting of this task. We also show the performance of AsSAYFORMER with and without context, illustrating its relatively poor ability without feedback but substantially stronger ICL capabilities.