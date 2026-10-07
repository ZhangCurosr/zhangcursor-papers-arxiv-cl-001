# CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling

Maoqi Liu<sup>1</sup>, Quan Fang<sup>1,B</sup>, Yufei He<sup>2</sup> <sup>1</sup>Beijing University of Posts and Telecommunications <sup>2</sup>National University of Singapore

## Abstract

Continual learning (CL) is essential for Large Language Models (LLMs) to sequentially adapt to evolving tasks. To mitigate catastrophic forgetting, recent advances implement low-rank adaptation with orthogonal projections (e.g., O-LoRA) to isolate task parameters. However, we reveal that such strict geometric constraints trigger an “Orthogonality Dilemma”: rigid parameter isolation impedes the transfer and accumulation of shared representations across semantically related tasks. In this work, we propose a new replay-free method, called Consolidation and Decoupling LoRA (CoDe-LoRA), for CL of LLMs. CoDe-LoRA disentangles the learning process into Consolidating Universal Knowledge and Decoupling Task-Specific Knowledge. To achieve this, CoDe-LoRA leverages an adaptive null space projection mechanism and semantic routing to balance knowledge accumulation with task-specific adaptation. Experimental results across four backbones and three CL benchmarks show that CoDe-LoRA achieves the best average accuracy. Our code is available at https: //github.com/Estrellajer/CoDe-LoRA.

## 1 Introduction

Large Language Models (LLMs) have redefined the boundaries of natural language processing, exhibiting remarkable capabilities across a wide range of tasks (Wei et al., 2021). However, adapting these models to a continuous stream of evolving tasks, known as Continual Learning (CL) (De Lange et al., 2021), remains challenging. While parametereficient fine-tuning (PEFT) methods, particularly low-rank adaptation (LoRA) (Hu et al., 2021), have significantly lowered the cost of adaptation, naively applying them to sequential tasks leads to catastrophic forgetting (Li and Hoiem, 2017), where learning new tasks degrades previously acquired capabilities on earlier tasks.

To mitigate forgetting, recent work combines PEFT with orthogonal projection strategies (Saha et al., 2021; Wang et al., 2023a). A representative example is O-LoRA (Wang et al., 2023a), which isolates task-specific parameters in mutually orthogonal subspaces. Although efective at reducing inter-task interference, such rigid isolation introduces a trade-of between interference mitigation and knowledge transfer. When sequential tasks share semantic structure, their optimal parameter updates should naturally overlap. Enforcing strict orthogonality removes these shared components, reducing representation reuse and weakening crosstask transfer. As illustrated in Figure 1, this phenomenon reveals an “Orthogonality Dilemma”: the same geometric constraint that protects previous knowledge can also restrict transferable representation learning across related tasks.

This limitation highlights a key challenge in continual adaptation: how to preserve prior knowledge while still accumulating transferable representations across related tasks. Motivated by this tension, we investigate two questions:

Q1. Why does strict orthogonality limit generalization across related tasks, and under what conditions is this limitation most severe?

Q2. How can we design a parameter-eficient framework that preserves transferable representations while maintaining task-specific expertise in a replay-free setting?

In this paper, we introduce Consolidation and Decoupling LoRA (CoDe-LoRA), a dual-branch continual adaptation framework motivated by the orthogonality-induced transfer limitation. We provide a geometric characterization showing that strict orthogonal isolation becomes increasingly detrimental as task similarity grows.

CoDe-LoRA separates transferable representation consolidation from task-specific specialization through two complementary branches. Unlike strict orthogonal isolation methods that suppress overlapping representations entirely, our framework preserves transferable components while constraining interference-prone directions. Specifically, the Consolidation Branch (Co-LoRA) maintains shared transferable representations through adaptive nullspace projection with dynamic scaling to stabilize long-horizon continual adaptation, as empirically verified in Appendix A.2 (Table 5). Meanwhile, the Decoupling Branch (De-LoRA) maintains a pool of independent LoRA experts together with a prototypebased semantic routing mechanism for robust inference without explicit task identifiers. Importantly, CoDe-LoRA operates without relying on historical data replay, avoiding additional storage overhead and privacy concerns.

![](images/99e4bd6df5c5894ec270c58eb229aa657621d59111742fd3e401322d6242c582.jpg)  
Figure 1. Conceptual illustration of parameter space utilization. Traditional LoRA sufers from interference, while O-LoRA’s strict orthogonality hinders knowledge sharing. CoDe-LoRA leverages null space projection to consolidate shared knowledge.

Our main contributions are summarized as follows:

• We identify and analyze the Orthogonality Dilemma in PEFT-based continual learning, showing that strict parameter isolation becomes increasingly detrimental as task similarity grows.

• We propose CoDe-LoRA, a dual-branch continual adaptation framework that separates transferable representation consolidation from taskspecific specialization through adaptive nullspace projection and prototype-based expert decoupling.

• Extensive experiments across four backbones and three continual learning benchmarks show that CoDe-LoRA achieves the best average accuracy while balancing knowledge transfer and interference mitigation.

## 2 Related Work

## 2.1 Parameter-Eficient Fine-Tuning

Parameter-Eficient Fine-Tuning (PEFT) adapts Large Language Models (LLMs) with minimal additional parameters. Early approaches such as Adapters (Houlsby et al., 2019) and Prompt/Prefix

Tuning (Li and Liang, 2021; Lester et al., 2021) introduced lightweight modules while freezing the backbone. LoRA (Hu et al., 2021) further improves eficiency through low-rank matrices. Although effective for static tasks, sequential adaptation with PEFT methods often leads to catastrophic forgetting of earlier tasks as training proceeds.

## 2.2 Continual Fine-Tuning for LLMs

Continual fine-tuning methods mainly rely on regularization (Kirkpatrick et al., 2017), replay (Qin and Joty, 2021; He et al., 2024), or parameter expansion (Razdaibiedina et al., 2023). Expansion-based methods avoid interference through task-specific parameters, but their parameter cost grows with the number of tasks. Replay and regularization methods preserve prior knowledge through memory bufers or parameter constraints, yet often incur computational overhead or privacy concerns. Recently, mixture-of-experts (MoE) routing frameworks such as MoLE-CIE (Wang et al., 2025) have been introduced for continual adaptation; however, they typically still rely on historical exemplar replay to prevent routing drift and maintain stable performance over long task sequences and domains.

More recently, orthogonality-based methods such as O-LoRA (Wang et al., 2023a), CLoRA (Lu et al., 2025), and N-LoRA (Yang et al., 2025b) mitigate forgetting by projecting updates into orthogonal or null subspaces. Although efective at reducing interference, strict orthogonal isolation may limit representation reuse across related tasks. In contrast, CoDe-LoRA separates shared representation consolidation from task-specific specialization, balancing knowledge transfer and interference mitigation through adaptive null-space projection without historical data replay.

![](images/380360fecf7cc32eeddc3a56d1c33f23d730bab433a0902de9884b2f2b9c020f.jpg)  
Figure 2. Preliminary evidence for the orthogonality dilemma and transferable representation accumulation. (a) Orthogonality-induced degradation under diferent task similarity levels. (b) Transfer trajectories over repeated training with and without orthogonal constraints. (c) Next-task zero-shot transfer comparison across continual adaptation methods. (d) Forward transfer dynamics across sequential tasks.

## 3 Preliminary

## 3.1 Problem Definition

We consider continual learning (CL) over a sequence of tasks $\{ \mathcal { T } ^ { ( 1 ) } , \mathcal { T } ^ { ( 2 ) } , . . . , \mathcal { T } ^ { ( T ) } \}$ . Each task $\mathcal { T } ^ { ( t ) }$ is associated with a dataset $\mathcal { D } ^ { ( t ) } = \{ ( x , y ) \}$ 2 where � denotes the input prompt and � the target sequence. Under the PEFT setting, the backbone parameters $\theta _ { \mathrm { b a s e } }$ remain frozen while task-specific updates $\Delta \theta ^ { ( t ) }$ are learned with a small number of trainable parameters. We consider the replay-free setting, where no data from earlier tasks is stored, and task identity is not given at inference time. Sequentially fine-tuning on new tasks often leads to catastrophic forgetting of previous ones.

## 3.2 Orthogonal Continual Adaptation

To mitigate forgetting, O-LoRA (Wang et al., 2023a) constrains each task update to be orthogonal to previous task subspaces. For task �, the LoRA update is parameterized as $\Delta W _ { t } = B _ { t } A _ { t }$ , where $A _ { t } \in \mathbb { R } ^ { r \times d _ { \operatorname* { i n } } }$ and $B _ { t } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ . Orthogonality is imposed through:

$$
A _ { i } A _ { t } ^ { \top } = 0 , \quad \forall i < t\tag{1}
$$

In practice it is enforced by a regularization penalty.

## 3.3 Orthogonality Dilemma

Although orthogonal constraints reduce interference, they may also suppress transferable representations across closely related tasks. Let $\Delta W _ { t } ^ { * }$ be the update that unconstrained fine-tuning learns on task �, and let $P _ { < t }$ be the orthogonal projector onto the row space of $A _ { < t } = [ A _ { 1 } ; \dots ; A _ { t - 1 } ]$ . We measure how much of this update is aligned with earlier tasks by

$$
\rho _ { t } = \frac { \| \Delta W _ { t } ^ { * } P _ { < t } \| _ { F } ^ { 2 } } { \| \Delta W _ { t } ^ { * } \| _ { F } ^ { 2 } } \in [ 0 , 1 ] .\tag{2}
$$

Enforcing $A _ { i } A _ { t } ^ { \top } = 0$ for all $i < t$ forces $\Delta W _ { t } P _ { < t } = 0$ , so the closest feasible update is $\Delta W _ { t } ^ { * } ( I - P _ { < t } )$ , which discards exactly $\mathbf { a } \rho _ { t }$ fraction of the update energy. The more related the tasks, the more reusable knowledge is lost.

This creates an “Orthogonality Dilemma”: the constraint that protects previous knowledge also limits transfer between related tasks.

## 3.4 Preliminary Evidence and Mechanism Validation

Figure 2 demonstrates these dynamics: (a) orthogonality-induced degradation worsens as task similarity grows; (b) unconstrained fine-tuning accumulates transferable knowledge over iterations, whereas strict orthogonality suppresses it; (c) preserving shared representations improves zero-shot transfer; and (d) transfer gains remain positive across task sequences.

Mechanism Validation. To investigate the underlying cause, we define task similarity as the cosine between task centroids, sim $( \mathcal T ^ { ( i ) } , \mathcal T ^ { ( j ) } ) = c ( \mathcal T ^ { ( i ) } )$ $c ( \mathcal { T } ^ { \left( j \right) } )$ , where $c ( \mathcal { T } )$ is the L2-normalized mean of mask-pooled, L2-normalized frozen T5-large embeddings of 200 training inputs, and measure update overlap by $\rho _ { t }$ (Eq. 2) on each task’s saved low-rank update. We compare O-LoRA (rank 10) with and without its orthogonality penalty, holding the Order-1 data, schedule, and seed fixed. The penalty $( \lambda = 0 . 5 )$ suppresses the prior-subspace energy from 5.05–5.63% to 0.47–0.64% (89.0% on average), while removing it $\left( \lambda = 0 \right)$ raises the mean next-task forward transfer from 43.70 to 49.45 (+5.75). This confirms that strict orthogonality suppresses update components that contribute to transfer, motivating our dual-branch design.

![](images/e4f5ad8d3561ed86a892788e228a7dbb96f8f297f3249c1e46a9522a68ef9058.jpg)  
Figure 3. The CoDe-LoRA framework. (Left) Training: Knowledge is disentangled into task-specific experts and a shared branch consolidated via null space projection. (Right) Inference: Inputs are routed to experts via prototype similarity, with a confidence-based fallback to the shared branch for robust prediction.

## 4 Methodology

## 4.1 Overall Framework

To resolve the orthogonality dilemma (Section 3.3), we propose CoDe-LoRA (illustrated in Figure 3), a dual-branch framework designed to consolidate shared representations while decoupling taskspecific expertise. Built upon a frozen backbone $\theta _ { \mathrm { b a s e } ; }$ , our architecture operates through two parallel pathways: the Consolidation Branch (Co-LoRA, $\theta _ { \mathrm { s h a r e d } } )$ , which accumulates transferable representations across tasks via null space projection (Sec. 4.2), and the Decoupling Branch (De-LoRA), which maintains a pool of independent task-specific LoRA experts $\{ \theta _ { \mathrm { s p e c } } ^ { k } \} _ { k = 1 } ^ { T }$ that captures domain nuances without interference (Sec. 4.3). These components are ultimately integrated via a confidencebased fallback selection strategy (Sec. 4.4) to ensure robust inference.

## 4.2 Consolidating Shared Representations

While strict parameter isolation (e.g., O-LoRA) prevents catastrophic forgetting, it often hinders the accumulation of shared representations across closely related tasks. Unlike strict orthogonal isolation methods that entirely discard overlapping update components, our consolidation branch preserves accumulated shared representations while constraining only newly introduced conflicting update directions. To achieve this objective, we propose a consolidation mechanism based on null space projection and dynamic scaling.

Consolidation Algorithm. After training on each new task $\mathcal { T } ^ { ( t ) }$ , we update the parameters of the Consolidation Branch (Co-LoRA) through a three-step consolidation process. The Consolidation Branch is parameterized by low-rank weights $W _ { \mathrm { s h a r e d } } ~ =$ $B _ { \mathrm { s h a r e d } } A _ { \mathrm { s h a r e d } }$ . Let ${ \dot { W _ { \mathrm { a c c } } ^ { ( t - 1 ) } } }$ denote the accumulated shared weight matrix of Co-LoRA up to task � − 1, and let $\Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) }$ represent the parameter update learned in the Consolidation Branch during the training of task �.

Null Space Projection. We project $\Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) }$ onto the null space of the existing representations represented by $W _ { \mathrm { a c c } } ^ { ( t - 1 ) }$ . Specifically, we perform Singular Value Decomposition (SVD) on $W _ { \mathrm { a c c } } ^ { ( t - 1 ) } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ to obtain its top-� left singular vectors $U \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \boldsymbol { v } } }$ which form an orthonormal basis for its column space. Here, $d _ { \nu } \leq r$ is the rank of the accumulated column space (we use $d _ { \nu } = r )$ . The new component $\Delta W _ { \mathrm { n u l l } }$ is then computed by removing the projection of $\Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) }$ onto this subspace:

$$
\Delta W _ { \mathrm { n u l l } } = \Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) } - U U ^ { \top } \Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) }\tag{3}
$$

Intuitively, the projection keeps only what is new in each task, so earlier representations are not overwritten. This projection reduces interference with previously consolidated representations. To improve scalability for large LLMs, we implement the projection using randomized low-rank SVD directly on factorized LoRA matrices. Details and eficiency analysis are provided in Appendix A.3.

Dynamic Scaling Constraint. To maintain the stability of the parameter magnitude, we introduce time-varying scaling coeficients $c _ { t } ~ = ~ \sqrt { \frac { t - 1 } { t } }$ and $\begin{array} { r } { s _ { t } = \frac { 1 } { \sqrt { t } } } \end{array}$ satisfying the normalization identity $c _ { t } ^ { 2 } + s _ { t } ^ { 2 } = \mathrm { \ i }$ . Since the null space projection ensures that each new update is orthogonal to the existing parameters, we scale the null space update component by $s _ { t }$ . This constraint prevents the cumulative parameter magnitude from exploding or drifting as the number of sequential tasks increases. We provide the theoretical proof of the stable parameter bound and its empirical verification in Appendix A.2.

The final consolidated weight matrix is updated by merging the scaled null space component with the previous state:

$$
W _ { \mathrm { a c c } } ^ { ( t ) } = c _ { t } W _ { \mathrm { a c c } } ^ { ( t - 1 ) } + s _ { t } \Delta W _ { \mathrm { n u l l } }\tag{4}
$$

This recursive formulation ensures that the consolidated weight matrix represents a normalized accumulation of orthogonal task updates. The choice of scaling coeficients stabilizes the parameter magnitude across tasks as follows:

Proposition 1. Given the recursive update rule defined in Equation (4), assuming the initial LoRA weights are zero $( W _ { a c c } ^ { ( 0 ) } ~ = ~ W ^ { ( 0 ) } ~ = ~ 0 )$ and that � spans the column space of $W _ { a c c } ^ { ( t - 1 ) }$ , the distance between the consolidated model and the pre-trained model remains bounded:

$$
\left\| W _ { a c c } ^ { ( t ) } - W ^ { ( 0 ) } \right\| _ { F } ^ { 2 } \leq M _ { t }\tag{5}
$$

where $M _ { t } = \operatorname* { m a x } _ { i \in [ 1 , t ] } \Big \| \Delta W _ { s h a r e d } ^ { ( i ) } \Big \| _ { F } ^ { 2 } .$

## 4.3 Decoupling Task-Specific Expertise

While the Consolidation Branch captures transferable representations, task-specific nuances are isolated within the Decoupling Branch (De-LoRA), consisting of a pool of task-specific experts $\{ \theta _ { \mathrm { s p e c } } ^ { k } \} _ { k = 1 } ^ { T }$ . To enable task-agnostic inference, we introduce a semantic routing mechanism that leverages task prototypes to dynamically activate the most relevant De-LoRA expert for each input. Prototype Construction. For each task, we construct a prototype $P _ { k } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ as the normalized mean of the frozen-backbone embeddings $f _ { \theta _ { \mathrm { b a s e } } } ( x _ { i } )$ of � randomly selected training samples (fixed to $N = 1 0$ throughout experiments). Crucially, because prototypes and runtime query representations are extracted solely using the frozen base model $\theta _ { \mathrm { b a s e } }$ rather than adapter-modulated layers, the semantic matching space does not drift over sequential learning phases.

Routing Procedure. During inference, the system identifies the optimal De-LoRA expert $\theta _ { \mathrm { s p e c } } ^ { k ^ { \ast } }$ for a given input � by measuring the similarity between the input embedding $e _ { x } = f _ { \theta _ { \mathrm { b a s e } } } ( x )$ and the stored prototypes $\{ P _ { k } \} _ { k = 1 } ^ { T }$ . Since prototypes are prenormalized, the routing index $\begin{array} { r } { k ^ { * } = \arg \operatorname* { m a x } _ { k } \frac { e _ { x } \cdot P _ { k } } { \| e _ { x } \| } } \end{array}$ is the prototype with the highest cosine similarity. Routing stores only compact per-task statistics rather than raw samples, so it remains replayfree. The selected expert $\theta _ { \mathrm { s p e c } } ^ { k ^ { * } }$ is then activated and integrated with the Consolidation Branch for inference via the fallback strategy described in Section 4.4. This similarity-based routing facilitates positive forward transfer: for inputs that bridge multiple task distributions, the mechanism naturally selects the most semantically related expert, enabling the model to leverage relevant prior knowledge. Appendix A.5 compares the routers and the cost of routing errors.

## 4.4 Knowledge Integration and Fallback Selection

To produce the final prediction, we employ a confidence-based fallback selection strategy that prioritizes specialized expertise while leveraging the consolidated shared branch as a robust prior. Here, the routing confidence is defined as the maximum cosine similarity between the input embedding and task prototypes. If the routing confidence exceeds a threshold � (a hyperparameter set to 0.75 by default), the system routes inference to the task-specific expert $\theta _ { \mathrm { s p e c } } ^ { k ^ { * } } ,$ executing $\hat { y } = \mathrm { M o d e l } ( x ; \theta _ { \mathrm { b a s e } } , \theta _ { \mathrm { s p e c } } ^ { k ^ { \ast } } )$ Otherwise, the model falls back to the consolidated shared branch, executing $\hat { y } = \mathrm { M o d e l } ( x ; \theta _ { \mathrm { b a s e } } , \theta _ { \mathrm { s h a r e d } } )$ . A higher � trusts the router only on clear matches and sends the rest to the shared branch, whereas $\tau = 0$ always trusts the routed expert. This fallback strategy dynamically balances stability and plasticity by preserving high-capacity specialized expertise while utilizing the consolidated branch as a robust safety net for ambiguous or out-of-distribution inputs. For classification tasks, candidate-label scores are additionally PMI-calibrated. The complete training and consolidation workflow is detailed in Appendix A.3. Both branches share the same frozen backbone, so no backbone parameters are updated.

Table 1. Performance comparison of diferent methods using the T5 model on Standard CL Benchmark and Large Number of Tasks. The average accuracy after training on the final task is reported. The top two results (excluding the oracle and MTL references) are bolded and underlined. Subscripts denote standard deviations across 3 random seeds.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="4">Standard CL Benchmark</td><td colspan="4">Large Number of Tasks</td></tr><tr><td>Order-1</td><td>Order-2</td><td>Order-3</td><td>Avg</td><td>Order-4</td><td>Order-5</td><td>Order-6</td><td>Avg</td></tr><tr><td>Rr</td><td>Per-LoRA (Oracle)</td><td>81.4</td><td>81.3</td><td>81.4</td><td>81.4</td><td>77.6</td><td>77.4</td><td>77.5</td><td>77.5</td></tr><tr><td rowspan="2"></td><td>MTL</td><td>80.0</td><td>80.0</td><td>80.0</td><td>80.0</td><td>76.5</td><td>76.5</td><td>76.5</td><td>76.5</td></tr><tr><td>SeqFT</td><td>18.9</td><td>24.9</td><td>41.7</td><td>28.5</td><td>7.4</td><td>7.4</td><td>7.5</td><td>7.4</td></tr><tr><td>SeT</td><td>SeqLoRA</td><td>44.6</td><td>32.7</td><td>53.7</td><td>43.7</td><td>0.6</td><td>1.9</td><td>1.6</td><td>1.4</td></tr><tr><td rowspan="5">Cadasi cl</td><td>IncLoRA</td><td>66.0</td><td>64.9</td><td>68.3</td><td>66.4</td><td>63.3</td><td>58.5</td><td>61.7</td><td>61.2</td></tr><tr><td>Replay</td><td>55.2</td><td>56.9</td><td>61.3</td><td>57.8</td><td>55.0</td><td>54.6</td><td>53.1</td><td>54.2</td></tr><tr><td>EWC</td><td>48.7</td><td>47.7</td><td>54.5</td><td>50.3</td><td>45.3</td><td>44.5</td><td>45.6</td><td>45.1</td></tr><tr><td>LwF</td><td>54.4</td><td>53.1</td><td>49.6</td><td>52.3</td><td>50.1</td><td>43.1</td><td>47.4</td><td>46.9</td></tr><tr><td>L2P</td><td>60.3</td><td>61.7</td><td>61.1</td><td>61.0</td><td>57.5</td><td>53.8</td><td>56.9</td><td>56.1</td></tr><tr><td rowspan="7">Ada ancct</td><td>LFPT5</td><td>67.6</td><td>72.6</td><td>77.9</td><td>72.7</td><td>70.4</td><td>68.2</td><td>69.1</td><td>69.2</td></tr><tr><td>O-LoRA</td><td>75.4</td><td>75.7</td><td>76.3</td><td>75.8</td><td>72.3</td><td>64.8</td><td>71.6</td><td>69.6</td></tr><tr><td>MIGU</td><td>77.1</td><td>77.0</td><td>75.6</td><td>76.6</td><td>67.3</td><td>68.5</td><td>74.2</td><td>70.0</td></tr><tr><td>N-LoRA</td><td>79.2</td><td>78.4</td><td>78.8</td><td>78.8</td><td>73.6</td><td>70.3</td><td>73.2</td><td>72.4</td></tr><tr><td>CLoRA</td><td>79.7</td><td>79.1</td><td>78.2</td><td>79.0</td><td>70.7</td><td>65.6</td><td>68.2</td><td>68.1</td></tr><tr><td>Co-LoRA</td><td>75.9±0.2</td><td>75.5±0.2</td><td>75.3±0.7</td><td>75.6±0.4</td><td>64.0±1.7</td><td>67.4±1.3</td><td>63.6±1.8</td><td>65.0±1.9</td></tr><tr><td>De-LoRA CoDe-LoRA (Ours)</td><td>79.4±0.3 80.3±0.2</td><td>79.2±0.1 80.4±0.2</td><td>79.3±0.1 80.5±0.3</td><td>79.3±0.2 80.4±0.2</td><td>76.7±0.4 78.6±0.5</td><td>76.6±0.8 78.8±0.7</td><td>76.2±0.8 79.8±0.3</td><td>76.5±0.7 79.1±0.7</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We evaluate CoDe-LoRA on three benchmarks: the Standard CL Benchmark (Zhang et al., 2015), the Large Number of Tasks Benchmark (Razdaibiedina et al., 2023), and the TRACE benchmark (Wang et al., 2023b) which contains diverse generative tasks. Following Qin and Joty (2021), we evaluate across three distinct task orders for each benchmark to ensure robustness. The complete specifications, including task sequences, dataset statistics, and natural language instructions, are detailed in Appendix A.8 and A.9.

Evaluation Metrics. Let $a _ { i , j }$ denote the performance on task � after training on task �. To assess continual learning performance, we report two primary metrics: (1) Average Performance (AP), computed as the average task performance at the end of the continual sequence (Average Accuracy for classification benchmarks and Overall Performance for generative benchmarks); and (2) Backward Transfer (BWT), which measures the retention of previously learned knowledge across tasks. We additionally analyze forward transfer behavior to evaluate the model’s ability to accumulate transferable representations across sequential tasks. Formal definitions are provided in Appendix A.4. Baselines. We compare CoDe-LoRA against four categories of continual learning baselines: (1) upper-reference baselines, including Multi-Task Learning (MTL) and Per-LoRA (Oracle); (2) sequential fine-tuning methods, including SeqFT (de Masson D’Autume et al., 2019), SeqLoRA, and IncLoRA; (3) continual learning methods, including EWC (Kirkpatrick et al., 2017), LwF (Li and Hoiem, 2017), Replay (Lopez-Paz and Ranzato, 2017), L2P (Wang et al., 2022), and LFPT5 (Qin and Joty, 2021); and (4) parameter-eficient continual learning methods, including O-LoRA (Wang et al., 2023a), MIGU (Du et al., 2024), N-LoRA (Yang et al., 2025b), CLoRA (Lu et al., 2025), and the routing-based MoLE-CIE (Wang et al., 2025).

Implementation Details. We evaluate CoDe-LoRA across diferent model families and scales using T5 (Rafel et al., 2020), Llama 2-7B (Touvron et al., 2023), Qwen3-0.6B (Yang et al., 2025a), and Qwen3.5-4B (Team, 2026). All models are optimized with AdamW (Loshchilov and Hutter, 2017), training each task for one epoch with LoRA rank 8. The learning rate is 10<sup>−3</sup> for T5 and 10<sup>−4</sup> for Llama 2-7B and Qwen3.5-4B; on Qwen3-0.6B, the pertask-expert methods (De-LoRA, CoDe-LoRA) use $1 0 ^ { - 3 }$ and the single-shared-adapter methods use $1 0 ^ { - 4 }$ . Results are averaged over three independent runs. Hyperparameters, routing details, and implementation settings are provided in Appendix A.3.

Table 2. Results across backbones. Std./Long: average accuracy on the Standard CL/Long Sequence benchmarks; OP/BWT: Trace metrics. Co/De-LoRA: single-branch ablations. Top two per column are bolded and underlined.
<table><tr><td rowspan="2">Method</td><td colspan="4">Llama-2-7B</td><td colspan="4">Qwen3-0.6B</td><td colspan="4">Qwen3.5-4B</td></tr><tr><td>Std.</td><td>Long</td><td>OP</td><td>BWT</td><td>Std.</td><td>Long</td><td>OP</td><td>BWT</td><td>Std.</td><td>Long</td><td>OP</td><td>BWT</td></tr><tr><td>CLoRA</td><td>78.0</td><td>67.8</td><td>45.3</td><td>-6.0</td><td>68.0</td><td>61.6</td><td>39.8</td><td>-10.6</td><td>80.7</td><td>82.1</td><td>60.8</td><td>-4.0</td></tr><tr><td>O-LoRA</td><td>77.0</td><td>67.8</td><td>44.2</td><td>-3.6</td><td>70.7</td><td>64.0</td><td>41.3</td><td>-5.2</td><td>80.4</td><td>83.3</td><td>60.2</td><td>-1.3</td></tr><tr><td>N-LoRA</td><td>73.9</td><td>57.4</td><td>44.0</td><td>-1.6</td><td>65.9</td><td>58.9</td><td>41.8</td><td>-4.3</td><td>79.0</td><td>81.6</td><td>58.4</td><td>-1.4</td></tr><tr><td>MoLE-CIE</td><td>68.1</td><td>70.6</td><td>36.2</td><td>-17.2</td><td>69.5</td><td>69.3</td><td>43.0</td><td>-6.1</td><td>70.9</td><td>78.4</td><td>49.1</td><td>-17.8</td></tr><tr><td>Co-LoRA</td><td>73.0</td><td>59.6</td><td>43.3</td><td>-2.0</td><td>66.4</td><td>60.1</td><td>41.3</td><td>-4.6</td><td>78.9</td><td>77.5</td><td>56.8</td><td>-2.3</td></tr><tr><td>De-LoRA</td><td>81.1</td><td>73.7</td><td>50.0</td><td>0.0</td><td>77.3</td><td>70.2</td><td>52.2</td><td>0.0</td><td>81.7</td><td>84.9</td><td>64.2</td><td>0.0</td></tr><tr><td>CoDe-LoRA</td><td>81.8</td><td>74.4</td><td>50.1</td><td>0.0</td><td>79.7</td><td>73.0</td><td>51.8</td><td>0.0</td><td>82.1</td><td>85.1</td><td>64.0</td><td>0.0</td></tr></table>

![](images/13a00d87c1413ccf6ccdda45839a9b830aedc254fd8fb941f4ab4c0dad40d2bf.jpg)  
(a) Task Order 1

![](images/6885a2e6be8add2b17aefc344c8969478409445b9b9765d8919263b515784d9f.jpg)  
(b) Task Order 2

![](images/84d41f1a8d94a8f60241ce39e391ac8f26c3149bf34537bde38252b6984946aa.jpg)  
(c) Task Order 3

![](images/1a742b22760370a4ec10b93d6eb7686667e8e280acc9129ba7aaac744fbe4b74.jpg)  
(d) Task Order 4

![](images/6b0f807e9ad8fcfb989dee405c1f7ad8937f885dedb31d8a6cab8ae3851d83de.jpg)  
(e) Task Order 5

![](images/0d2a7989daf5e12e656476746d484691b1b017bec010b7efc836e555e2e779aa.jpg)  
(f) Task Order 6  
Figure 5. Performance curves of diferent methods across six task orders on the Standard CL Benchmark (Orders 1-3) and Large Number of Tasks Benchmark (Orders 4-6). Results are averaged over three independent runs.

## 5.2 Main Results

On the Standard CL Benchmark (Table 1), CoDe-LoRA achieves the highest average accuracy (80.4%), outperforming prior PEFT-based CL methods and exceeding the MTL reference (80.0%). This result suggests that the decoupled expert pool provides task-specific capacity while avoiding the forgetting typical of sequential fine-tuning.

On the Large Number of Tasks benchmark, the performance gap is more pronounced: CoDe-LoRA achieves 79.1% average accuracy, outperforming the strongest baseline, N-LoRA, by 6.7 points (79.1% vs. 72.4%). While O-LoRA reduces interference through strict orthogonal isolation, its performance degrades on longer sequences where transferable representations become increasingly important. These results indicate stronger scalability in long-horizon scenarios.

Generalization to Generative Tasks. We further evaluate CoDe-LoRA on the TRACE benchmark (Wang et al., 2023b), which contains generative tasks. As shown in Table 2, CoDe-LoRA achieves competitive generation quality (OP) and consistently improves knowledge retention (BWT) without relying on replay bufers, indicating improved retention in generative domains.

Generalization across Model Families. To evaluate generalization across model architectures and scales, we evaluate CoDe-LoRA on Qwen3-0.6B, Qwen3.5-4B, and Llama 2-7B against isolationbased PEFT (O-LoRA, N-LoRA) and routing-based MoE (MoLE-CIE (Wang et al., 2025)) baselines (Table 2). CoDe-LoRA outperforms every baseline on every backbone and benchmark. On Qwen3-0.6B it improves over the strongest baseline by 9.0 points on standard CL (79.7 vs. 70.7) and by 3.7 points on the long-sequence benchmark (73.0 vs. 69.3); the corresponding margins are 3.8 and 3.8 on Llama 2- 7B and 1.4 and 1.8 on Qwen3.5-4B, and TRACE OP improves by 3.2 to 8.8 points. Because every task retains its own frozen expert, both De-LoRA and CoDe-LoRA show no measurable backward transfer loss on TRACE (BWT = 0.0 on all three backbones), whereas the strongest baseline per backbone still loses between 1.3 and 4.3 points, and MoLE-CIE loses up to 17.8. The single-branch ablations locate the gain: Co-LoRA alone is the weakest of the three, De-LoRA already accounts for most of the accuracy, and adding the consolidation branch adds a further 0.4 to 2.8 points on Std. CL and Long Seq. On TRACE, where the two branches cannot be combined, CoDe-LoRA matches De-LoRA. All of this is achieved without maintaining historical replay bufers.

## 5.3 Ablation Studies and Analysis

Capacity Control. We re-train O-LoRA and N-LoRA with � = 10 on T5-Large so that their per-task adapter footprint matches CoDe-LoRA at � = 8, keeping every other setting identical (three task orders, three seeds each). Raising the baseline rank changes little: O-LoRA moves from 76.5 to 76.4 on the Standard CL Benchmark and from 70.2 to 70.4 on the Long Sequence Benchmark, and N-LoRA does not improve either. Against the strongest capacity-matched baseline, CoDe-LoRA leads by 4.0 points (80.4 vs. 76.4) and by 8.7 points (79.1 vs. 70.4), respectively. The gains therefore stem from decoupled knowledge consolidation rather than from parameter capacity.

Ensemble Strategy. We compare CoDe-LoRA with LoRAHub (Huang et al., 2023), which directly merges task-specific LoRA modules. As shown in Figure 6, LoRAHub exhibits severe catastrophic forgetting, yielding a BWT of -20.2% compared to -0.2% for CoDe-LoRA. This suggests that direct parameter merging introduces significantly stronger cross-task interference.

Table 3. Eficiency on Order 1 (T5-large, one H20 GPU). Train and inference are seconds per task and per evaluated test set; CoDe-LoRA is a single seed, the others are means over three seeds. Params are the stored adapter parameters (CoDe-LoRA stores only the experts) and their share of the backbone.
<table><tr><td>Method</td><td>Train (s) Infer (s)</td><td></td><td>Params</td></tr><tr><td>O-LoRA</td><td>108.8</td><td>30.1</td><td>9.4M (1.28%)</td></tr><tr><td>N-LoRA</td><td>98.0</td><td>29.9</td><td>9.4M (1.28%)</td></tr><tr><td>MoLE-CIE</td><td>861.0</td><td>142.7</td><td>32.4M (4.40%)</td></tr><tr><td>CoDe-LoRA</td><td>98.4</td><td>43.1</td><td>9.4M (1.28%)</td></tr></table>

Forward Transfer. Figure 7 shows the zero-shot forward transfer matrix on Qwen3-0.6B (Order 1). CoDe-LoRA achieves consistently higher forward transfer than O-LoRA, particularly after progressively accumulating knowledge across tasks (e.g., +19.1 points on Yahoo after training on dbpedia and amazon). This suggests that the consolidation branch efectively accumulates transferable representations that benefit future unseen tasks.

Impact of Prototype Samples. We analyze the sensitivity of routing performance to the number of prototype samples �. Figure 6 (right) shows that routing accuracy improves as � increases, with performance saturating around � = 10. This indicates that only a small number of samples are required for reliable task routing.

Contribution of Each Branch. To further analyze the necessity of the proposed dual-branch design, we compare the Consolidation Branch alone (Co-LoRA), the Decoupling Branch alone (De-LoRA), and the full CoDe-LoRA architecture on the Standard CL Benchmark, as reported in the corresponding rows of Table 1. The consolidation branch (Co-LoRA) alone captures shared representations, while the decoupling branch (De-LoRA) alone preserves task-specific adaptation. Combining both branches yields the best overall performance across all task sequences (Table 1). Appendix A.6 reports the complementarity of the two branches and the attempts to exploit it.

Eficiency and Scalability Analysis. CoDe-LoRA trains one adapter per task on top of the consolidated branch, so a task costs as much as for O-LoRA and N-LoRA (98.4 s vs. 108.8 s and 98.0 s on the

![](images/bf86426992a20a7f3cb3f36ee1d98b8eadb888612cf8cf4040536ebe93ab822b.jpg)

![](images/beacd1ba9dfe0d3ceb6c41d2055d43142f38945a70d8301f3ff0a56209d38f08.jpg)

![](images/1fd7c81c119be46e8cedce6d514af27dd79f04b0eb9620b738f010a80ab4c6a6.jpg)

![](images/ea2c517b251d0f8dad3be2e4099fb17bc427f5b3ebad2eecbc29b71f237fbac4.jpg)  
Figure 6. Ablation analysis of CoDe-LoRA showing task-wise accuracy and BWT under diferent ensemble strategies (left) and routing accuracy across varying prototype sample sizes � (right).

![](images/44391137482cdf841481aad8a5e0325f412ae05457a0e7316d1a3b238d4ce8d4.jpg)  
Figure 7. Zero-shot forward transfer on Qwen3-0.6B (Order 1). “N/A” indicates trained tasks.

Standard benchmark, 50.1 s vs. 53.2 s and 47.9 s on the Long benchmark) and about 9× less than MoLE-CIE (Table 3). It stores only the experts, the same 9.4M adapter parameters as O-LoRA and N-LoRA on four tasks (1.28% of the backbone, and 35.4M or 4.80% on the fifteen tasks of the Long benchmark; Appendix Table 11); the consolidated branch and the per-task contexts are rebuilt by replaying the folds. The price is inference: replaying the context of the routed expert makes an evaluation pass about 1.4× slower than O-LoRA (43.1 s vs. 30.1 s).

## 6 Conclusion

We revisit orthogonal continual adaptation from the perspective of transferable representation learning. We show that while strict orthogonal isolation mitigates catastrophic forgetting, it can also suppress transferable representation accumulation across related tasks. To address this limitation, we propose CoDe-LoRA, a dual-branch continual adaptation framework that balances shared representation consolidation and task-specific adaptation. Extensive experiments demonstrate strong continual learning performance together with robust forward and backward transfer across diverse task sequences and model families.

## Limitations

Although CoDe-LoRA demonstrates strong performance across both classification and generative continual learning benchmarks, our evaluation is limited to pre-trained language models. Extending the framework to multi-modal continual learning settings remains an important direction for future work. In addition, the efectiveness of the routing mechanism depends on the quality of task prototypes in the embedding space. Under severe task distribution shifts or highly overlapping task boundaries, routing reliability may degrade and require more adaptive prototype estimation strategies.

## Acknowledgements

This work was supported by Beijing Natural Science Foundation (JQ24019). This work was supported by the National Natural Science Foundation of China (No. 62576047). This work is being sponsored by CIPS-SMP-Zhipu Large Model Fund (No. CIPS-SMP20250313) and supported by the Open Fund of the Key Laboratory for Civil Aviation Collaborative Air Trafic Management Technology and Applications (No. 2025-001).

## References

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. 2021. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7):3366– 3385.

Cyprien de Masson D’Autume, Sebastian Ruder, Lingpeng Kong, and Dani Yogatama. 2019. Episodic memory in life-

long language learning. Advances in Neural Information Processing Systems, 32.

Wenyu Du, Shuang Cheng, Tongxu Luo, Zihan Qiu, Zeyu Huang, Ka Chun Cheung, Reynold Cheng, and Jie Fu. 2024. Unlocking continual learning abilities in language models. In Conference on Empirical Methods in Natural Language Processing.

Jinghan He, Haiyun Guo, Kuan Zhu, Zihan Zhao, Ming Tang, and Jinqiao Wang. 2024. Seekr: Selective attentionguided knowledge retention for continual learning of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3254–3266.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parametereficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. 2023. Lorahub: Eficient cross-task generalization via dynamic lora composition. Preprint, arXiv:2307.13269.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-eficient prompt tuning. arXiv preprint arXiv:2104.08691.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Opti mizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.

Zhizhong Li and Derek Hoiem. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Yuheng Lu, Bingshuo Qian, Caixia Yuan, Huixing Jiang, and Xiaojie Wang. 2025. Controlled low-rank adaptation with subspace regularization for continued training on large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19165–19181.

Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the

49th annual meeting of the association for computational linguistics: Human language technologies, pages 142–150.

Chengwei Qin and Shafiq Joty. 2021. Lfpt5: A unified framework for lifelong few-shot language learning based on prompt tuning of t5. arXiv preprint arXiv:2110.07298.

Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(1).

Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, and Amjad Almahairi. 2023. Progressive prompts: Continual learning for language models. arXiv preprint arXiv:2301.12314.

Gobinda Saha, Isha Garg, and Kaushik Roy. 2021. Gradient projection memory for continual learning. arXiv preprint arXiv:2103.09762.

Qwen Team. 2026. Qwen3.5: Accelerating productivity with native multimodal agents.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355. Association for Computational Linguistics.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. 2023a. Orthogonal subspace learning for language model continual learning. arXiv preprint arXiv:2310.14152.

Xiao Wang, Yuansen Zhang, Tianze Chen, Songyang Gao, Senjie Jin, Xianjun Yang, Zhiheng Xi, Rui Zheng, Yicheng Zou, Tao Gui, et al. 2023b. Trace: A comprehensive benchmark for continual learning in large language models. arXiv preprint arXiv:2310.06762.

Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. 2022. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 139–149.

Zitao Wang, Xinyi Wang, and Wei Hu. 2025. Mixture of LoRA experts for continual information extraction with LLMs. In Findings ofthe Associationfor Computational Lin-

guistics: EMNLP 2025, pages 13324–13339. Association for Computational Linguistics.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Shuo Yang, Kun-Peng Ning, Yu-Yang Liu, Jia-Yu Yao, Yong-Hong Tian, Yi-Bing Song, and Li Yuan. 2025b. Is parameter collision hindering continual learning in LLMs? In Proceedings of the 31st International Conference on Computational Linguistics, pages 4243–4259. Association for Computational Linguistics.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28.

## A Appendix

## A.1 Notation

We summarize the key notations used throughout this paper in Table 4.

Table 4. Summary of key notations.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\mathcal { T } ^ { ( t ) }$ </td><td>The t-th task in the sequence</td></tr><tr><td> $\mathcal { D } ^ { ( t ) }$ </td><td>Dataset associated with task  $\mathcal { T } ^ { ( t ) }$ </td></tr><tr><td> $\theta _ { \mathrm { b a s e } }$ </td><td>Parameters of the frozen backbone</td></tr><tr><td> $\theta _ { \mathrm { s h a r e d } }$ </td><td>Parameters of consolidation branch</td></tr><tr><td> $\theta _ { \mathrm { s p e c } } ^ { k }$ </td><td>Parameters of k-th task expert</td></tr><tr><td> $W _ { \mathrm { a c c } } ^ { ( t ) }$ </td><td>Accumulated shared weights up to task t</td></tr><tr><td> $\Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) }$ </td><td>Consolidation branch update at task t</td></tr><tr><td> $\Delta W _ { \mathrm { n u l l } }$ </td><td>Null space projected update component</td></tr><tr><td> $c _ { t } , s _ { t }$ </td><td>Dynamic scaling coefficients at task t</td></tr><tr><td> $P _ { k }$ </td><td>Semantic prototype for task k</td></tr><tr><td> $e _ { x }$ </td><td>Embedding of input x from backbone</td></tr><tr><td> $\tau$ </td><td>Fallback confidence threshold</td></tr></table>

## A.2 Theoretical Justification for Dynamic Scaling

In the model merging process, the dynamic timevarying scaling coeficients are introduced to maintain the deviation magnitude between the merged model and the pre-trained model.

General Formulation of Scaling. For clarity, we denote the parameter deviation at step � as $D _ { t } ~ = ~ W _ { \mathrm { a c c } } ^ { ( t ) } - W ^ { ( 0 ) }$ , the null-space projected update component as $V _ { t } ~ = ~ \Delta W _ { \mathrm { n u l l } } ^ { ( t ) } ~ = ~ P _ { \mathrm { n u l l } } \Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) } ,$ and the running average norm of task updates as $\begin{array} { r } { \mu _ { t } = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \| \Delta W _ { \mathrm { s h a r e d } } ^ { ( i ) } \| _ { F } } \end{array}$ . In our study, the generalized scaling factor $\lambda _ { t }$ is defined as the inverse of the scaling coeficient $s _ { t } ~ ( \mathrm { i . e . , } ~ \lambda _ { t } = 1 / s _ { t } )$ , and can be computed as:

$$
\lambda _ { t } = \lambda _ { t - 1 } \cdot \frac { \| D _ { t - 1 } + V _ { t } \| _ { F } } { \mu _ { t } }\tag{6}
$$

where $P _ { \mathrm { n u l l } }$ denotes the null space projection operator.

Derivation of Scaling Coeficients.

Proof of Proposition 1. To simplify notation in this proof, we denote the consolidated weight matrix $W _ { \mathrm { a c c } } ^ { ( t ) }$ as $W _ { t }$ , and the null-space projected update component $\Delta W _ { \mathrm { n u l l } } ^ { ( t ) }$ as $V _ { t }$ . Recall that we assume $W ^ { ( 0 ) } = W _ { 0 } = 0$ . Under the update rule defined in Eq. (4), the consolidated weight matrix at task � is updated recursively as:

$$
W _ { t } = c _ { t } W _ { t - 1 } + s _ { t } V _ { t }\tag{7}
$$

where the scaling coeficients $c _ { t } = \sqrt { ( t - 1 ) / t }$ and $s _ { t } = 1 / \sqrt { t }$ satisfy the normalization identity $c _ { t } ^ { 2 } + s _ { t } ^ { 2 } =$ 1. By definition, the projected component $V _ { t }$ lies in the null space of the accumulated representations $W _ { t - 1 }$ , which implies they are orthogonal under the Frobenius inner product (exactly when � spans the column space of $W _ { t - 1 }$ , approximately for a truncated basis):

$$
\langle W _ { t - 1 } , V _ { t } \rangle _ { F } = 0\tag{8}
$$

Using this orthogonality, and given the boundary condition $W _ { 0 } = 0$ , we have $\lVert W _ { t } - W _ { 0 } \rVert _ { F } ^ { 2 } = \lVert W _ { t } \rVert _ { F } ^ { 2 }$ Applying the Pythagorean theorem to the update rule in Eq. (7) yields:

$$
\| W _ { t } \| _ { F } ^ { 2 } = c _ { t } ^ { 2 } \| W _ { t - 1 } \| _ { F } ^ { 2 } + s _ { t } ^ { 2 } \| V _ { t } \| _ { F } ^ { 2 }\tag{9}
$$

Let $\begin{array} { r } { M _ { t } ~ = ~ \operatorname* { m a x } _ { i \in [ 1 , t ] } \| \Delta W _ { \mathrm { s h a r e d } } ^ { ( i ) } \| _ { F } ^ { 2 } } \end{array}$ . We prove that $\| W _ { t } \| _ { F } ^ { 2 } \le M _ { t }$ by induction on �.

For the base case $t = 1$ , since $c _ { 1 } = 0$ and $s _ { 1 } = 1$ ， we have:

$$
\| W _ { 1 } \| _ { F } ^ { 2 } = \| V _ { 1 } \| _ { F } ^ { 2 } \leq M _ { 1 }\tag{10}
$$

where the inequality holds because the projection operator is a contraction mapping, i.e., $\Vert V _ { t } \Vert _ { F } ~ \leq$ $\lVert \Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) } \rVert _ { F } ,$

Now, assume that the bound holds up to task �−1, i.e., $\| W _ { t - 1 } \| _ { F } ^ { 2 } \leq M _ { t - 1 }$ . For task $t ,$ substituting the induction hypothesis and the contraction property

into Eq. (9):

$$
\| W _ { t } \| _ { F } ^ { 2 } \leq c _ { t } ^ { 2 } M _ { t - 1 } + s _ { t } ^ { 2 } \| \Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) } \| _ { F } ^ { 2 }\tag{11}
$$

$$
\leq c _ { t } ^ { 2 } M _ { t } + s _ { t } ^ { 2 } M _ { t }\tag{12}
$$

$$
= ( c _ { t } ^ { 2 } + s _ { t } ^ { 2 } ) M _ { t } = M _ { t }\tag{13}
$$

This completes the induction, showing that the Frobenius norm of the accumulated parameter deviation is bounded by the maximum individual task update norm:

$$
\| W _ { \mathrm { a c c } } ^ { ( t ) } - W ^ { ( 0 ) } \| _ { F } ^ { 2 } \leq M _ { t }\tag{14}
$$

This bound prevents the parameters of the consolidated branch from exploding or drifting away from the pre-trained manifold, mitigating catastrophic forgetting without sacrificing stability. □

Remark on Knowledge Dilution. One might be concerned that the $s _ { t } ~ = ~ 1 / \sqrt { t }$ scaling in Eq. (4) dilutes the historical knowledge in the shared memory. We argue that this scaling acts as a variance normalization mechanism rather than information loss. Without scaling, the norm of the shared branch would grow linearly with $\sqrt { t }$ (assuming orthogonal updates), eventually overwhelming the task-specific branch. The $s _ { t } = 1 / \sqrt { t }$ factor ensures that the shared memory remains a stable “anchor” while the specific branch (Section 4.3) preserves the distinct decision boundaries for each task. This dual-branch design efectively decouples stability (via normalized shared memory) from plasticity (via preserved specific experts).

Empirical Verification of $1 / { \sqrt { t } }$ Scaling. To empirically verify the stabilizing efect of our derived $\bar { 1 } / \sqrt { t }$ scaling factor against standard linear averaging $( 1 / t )$ or no scaling (1), we conduct a comparative analysis on the Standard CL Benchmark (Order-1, T5-Large). Table 5 reports the final average accuracy and the Frobenius norm of the accumulated parameters $\lVert W _ { \mathrm { a c c } } ^ { ( t ) } - W ^ { ( 0 ) } \rVert _ { F }$ . Without scaling, the parameter norm explodes, pushing weights outside the LLM’s efective optimization manifold. Conversely, linear averaging (1/�) overly dilutes the historical knowledge. Only our derived $1 / { \sqrt { t } }$ factor maintains a stable norm and achieves the peak accuracy.

## A.3 Implementation Details

Following existing CL works (Qin and Joty, 2021; Wang et al., 2023a; Yang et al., 2025b), all methods are implemented with instruction tuning (Qin and Joty, 2021) and optimized with AdamW (Loshchilov and Hutter, 2017) $\begin{array} { r l } { ( \beta } & { { } = } \end{array}$ (0.9, 0.999), no weight decay, gradient clipping at 1.0) under a constant learning rate, in bfloat16. Every method trains each task for one epoch with an efective batch size of 64, and inputs are truncated to 512 source and 50 target tokens. The learning rate is set per backbone: $1 0 ^ { - 3 }$ for T5-large, and $1 0 ^ { - 4 }$ for Llama 2-7B and Qwen3.5-4B. On Qwen3- 0.6B the per-task-expert methods (De-LoRA, CoDe-LoRA) use $1 0 ^ { - 3 }$ , while the methods built on a single shared adapter use $1 0 ^ { - 4 }$ . For all LoRA-based architectures we set rank $r = 8 , \alpha = 3 2$ and dropout 0.1, targeting the query and value projections of every Transformer block. Routing uses an LDA classifier on frozen-backbone embeddings estimated from 1000 support samples per task (Appendix A.5), the consolidation uses a randomized low-rank SVD with retained rank 8, and classification tasks are scored over their candidate labels with PMI calibration while TRACE is evaluated by generation.

Table 5. Scaling factors on the Standard CL Benchmark (Order-1, T5-Large), measured on the null-space consolidation path where this rule is defined. “Avg Accuracy” is the full model; “Shared Branch” evaluates the consolidated branch alone, which is what the scaling factor controls. Three seeds.
<table><tr><td>Method</td><td> $c _ { t } , \ s _ { t }$ </td><td>Avg Accuracy</td><td>Shared Branch</td></tr><tr><td>No Scaling</td><td> $^ { 1 , ~ 1 }$ </td><td> $7 6 . 5 _ { \pm 0 . 5 }$ </td><td> $6 0 . 3 _ { \pm 1 . 3 }$ </td></tr><tr><td>Linear Avg</td><td> $\textstyle { \frac { t - 1 } { t } } , \ { \frac { 1 } { t } }$ </td><td> $7 7 . 5 _ { \pm 0 . 4 }$ </td><td> $7 3 . 6 _ { \pm 1 . 0 }$ </td></tr><tr><td>CoDe-LoRA</td><td> $\textstyle { \sqrt { \frac { \operatorname { t } - 1 } { \operatorname { t } } } } , \ { \frac { 1 } { \sqrt { \operatorname { t } } } }$ </td><td> $7 7 . 8 _ { \pm 0 . 4 }$ </td><td> $7 5 . 6 _ { \pm 0 . 6 }$ </td></tr></table>

All experiments use NVIDIA H20 or A100 (80 GB) GPUs; the eficiency measurements in Appendix A.7 are taken on a single GPU to avoid inter-GPU communication efects. Each experiment is repeated over three independent runs with diferent random seeds.

Learning rates. The learning rate is set per backbone because no single value works for all of them. $\mathrm { A t } 1 0 ^ { - 3 }$ the methods built on a single shared adapter (LoRA, CLoRA) diverge on Llama 2-7B and Qwen3.5-4B (gradient norms of 100–340), and on Qwen3-0.6B they forget catastrophically (LoRA on the Long benchmark reaches an AP of only 10.5 with a BWT of −62). $\mathrm { A t } ~ 1 0 ^ { - 4 }$ the per-task experts of De-LoRA and CoDe-LoRA cannot learn the smallest tasks (CB and COPA, 4–6 optimizer steps per epoch) on Qwen3-0.6B, and De-LoRA falls from 71.2 to 59.5.

Practical Implementation Details of SVD and Null Space Projection. To resolve the computational bottleneck of null space projection in largescale LLMs, we adopt an optimized consolidation strategy. Specifically, we replace full Singular Value Decomposition (SVD) with a randomized low-rank SVD (torch.svd<sup>\_</sup>lowrank) to eficiently extract singular vectors � from the column space of consolidated parameters. To further reduce complexity, the projection is performed directly on the lowrank factor $B \in \mathbb { R } ^ { d _ { o u t } \times r }$ rather than the full update $\Delta W \in \mathbb { R } ^ { d _ { o u t } \times d _ { i n } }$ . The orthogonalized factor is computed as $\begin{array} { r } { B _ { \mathrm { o r t h } } = B - U ( U ^ { \top } B ) } \end{array}$ , and the resulting update is $\Delta W _ { \mathrm { n u l l } } = B _ { \mathrm { o r t h } } A$ . This optimization reduces the projection matrix dimensions from $d _ { o u t } \times d _ { i n }$ to $d _ { o u t } \times r .$ , where $r ~ \ll ~ d _ { i n }$ , achieving significant speedup while preserving mathematical accuracy. In practice, we set the number of randomized SVD iterations (svd<sup>\_</sup>niter) to 1 to balance precision and performance.

Computational Complexity and SVD Eficiency Bridge. As introduced in Section 4.2, standard SVD on the full weight matrix update $\Delta W \in \mathbb { R } ^ { d _ { o u t } \times d _ { i n } }$ requires $O ( d _ { o u t } d _ { i n } \operatorname* { m i n } ( d _ { o u t } , d _ { i n } ) )$ operations, which is prohibitively expensive for large language models where $d _ { o u t } , d _ { i n }$ can scale to thousands or tens of thousands. In contrast, by projecting directly on the low-rank factor $B \in \mathbb { R } ^ { d _ { o u t } \times r }$ and leveraging the randomized low-rank SVD method (torch.svd<sup>\_</sup>lowrank), the computational complexity is reduced to $\mathcal { O } ( d _ { o u t } r ^ { 2 } )$ where � is the LoRA rank (typically $r \in [ 8 , 3 2 ] )$ ). Because $r \ll d _ { i n } , d _ { o u t }$ , this optimization achieves a speedup of several orders of magnitude. In practice, the SVD consolidation is negligible: over a whole T5-large run it takes $0 . 7 \mathrm { : }$ s for the four tasks of the Standard benchmark and 4.1 s for the fifteen tasks of the Long benchmark, while preserving the mathematical guarantee of null space projection. This provides a direct, highly eficient bridge between the theoretical null space projection in Section 4.2 and the practical scalability of our framework.

Sensitivity to Approximate Orthogonality. While randomized SVD inevitably introduces approximation noise, CoDe-LoRA remains theoretically and architecturally robust to this relaxation. Theoretically, our dynamic scaling factor dampens noise accumulation, bounding the approximation error. Architecturally, the Decoupling Branch serves as a safety net; if approximation noise compromises the shared branch, routing dynamically isolates predictions to task-specific experts. To verify this, we vary the SVD retained rank on the Order-1 benchmark (T5-large, three seeds). Table 6 reports the accuracy of the consolidation branch alone and of the full model. The consolidation branch

Table 6. SVD retained rank on the Standard CL Benchmark (Order 1, T5-large). Rank 8 is the default.
<table><tr><td>Retained rank</td><td>Co-LoRA alone</td><td>CoDe-LoRA</td></tr><tr><td>1</td><td>66.84</td><td>79.13</td></tr><tr><td>2</td><td>72.11</td><td>79.13</td></tr><tr><td>4</td><td>75.86</td><td>79.13</td></tr><tr><td>8</td><td>76.67</td><td>79.13</td></tr></table>

alone improves monotonically with the retained rank, since a lower rank discards more of the accumulated update. The full model does not move because the routed expert answers every input, so the shared branch does not enter the prediction. The retained rank therefore matters for the quality of the consolidated weights, not for the routed accuracy.

Step-by-Step Training and Consolidation Workflow. During the learning of each sequential task $\mathcal { T } _ { t } ,$ CoDe-LoRA decouples and consolidates knowledge through a structured step-by-step procedure:

1. Initialization: The Consolidation Branch parameters $\theta _ { \mathrm { s h a r e d } }$ are initialized from the previously consolidated state $W _ { \mathrm { a c c } } ^ { ( t - 1 ) }$ to enable immediate knowledge reuse and positive transfer, while a new task-specific expert $\theta _ { \mathrm { s p e c } } ^ { ( t ) }$ is initialized to zero.

2. Joint Training: Both branches are optimized jointly on the current task dataset $\mathcal { D } ^ { ( t ) }$ using instruction tuning. This joint optimization allows the Consolidation Branch to assimilate shared representation updates while the Decoupling Branch captures task-specific nuances.

3. Null Space Consolidation: Post-training, the parameter update learned in the Consolidation Branch, denoted as $\Delta W _ { \mathrm { s h a r e d } } ^ { ( t ) } = W _ { \mathrm { s h a r e d } } ^ { ( t ) } - W _ { \mathrm { a c c } } ^ { ( t - 1 ) }$ is extracted. We project this update onto the null space of the accumulated representation history $W _ { \mathrm { a c c } } ^ { ( t - 1 ) }$ using randomized low-rank SVD (as detailed above) to eliminate representational interference.

4. Dynamic Scaling & Archive: The projected update is scaled via our dynamic scaling coeficient $s _ { t } = 1 / \sqrt { t }$ to prevent parameter norm explosion, and merged into the consolidated weight matrix $W _ { \mathrm { a c c } } ^ { ( t ) }$ recursively using $c _ { t } = \sqrt { ( t - 1 ) / t }$ . Concurrently, the trained task-specific expert $\theta _ { \mathrm { s p e c } } ^ { ( t ) }$ is archived in the expert pool, and its prototype $P _ { t }$ is computed and stored.

Evaluation Protocol and Analysis Setup. For text classification tasks (e.g., Yahoo, DBpedia), candidate labels are obtained from task metadata and tokenized. The main performance results are reported using the generate() method; classification tasks are additionally scored over their candidate labels via forward(). For the ablation studies and prototype sensitivity analysis in Figure 6, we vary the number of support samples � to find the optimal balance between routing performance and inference speed.

## A.4 Evaluation Metrics

We evaluate sequential learning performance using two standard metrics in continual learning:

Average Performance (AP) represents the overall model capacity across all learned tasks after completing the continual learning sequence. For a sequence of � tasks, it is defined as:

$$
\mathsf { A P } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } a _ { i , T }\tag{15}
$$

where $a _ { i , T }$ denotes the evaluation performance (accuracy for classification benchmarks, or taskspecific generative metrics like ROUGE/Exact Match for generative benchmarks) on task � after training on the final task �. In classification benchmarks, this corresponds to the standard Average Accuracy $\left( A _ { T } \right)$ , while in generative benchmarks (such as TRACE), it is referred to as Overall Performance (OP).

Backward Transfer (BWT) (Lopez-Paz and Ranzato, 2017) measures the influence of learning subsequent tasks on the performance of previously acquired knowledge:

$$
\mathrm { B W T } = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } ( a _ { i , T } - a _ { i , i } )\tag{16}
$$

where $a _ { i , i }$ is the performance on task � immediately after training on task �. A negative BWT value indicates catastrophic forgetting (performance degradation), whereas a value closer to 0 or positive indicates stable knowledge retention or positive backward transfer (where learning new tasks improves performance on older tasks). By employing BWT as our unified metric for backward retention across all benchmarks, we avoid the redundancy of using separate forgetting indicators for classification and generative tasks.

Forward Transfer (FWT) measures the zero-shot accuracy on a task that has not been trained yet. After learning the first � tasks, $\mathrm { F W T } _ { k  j } = a _ { j , k }$ for $j > k ,$ which we compare with the accuracy of the untrained backbone and of the other methods on the same task.

## A.5 Router Choice

The main experiments route with a linear discriminant analysis (LDA) classifier on frozen-backbone sentence embeddings. For every task we store the mean embedding and one covariance matrix shared by all tasks, estimated from 1000 support samples per task, and an input goes to the task with the largest discriminant score. No raw sample is stored, so the method stays replay-free and the per-task storage is one vector, as for a prototype, plus a single shared matrix. The cosine prototype router of Section 4.3 is also implemented.

Table 7. Routing errors and their cost on T5-large (three seeds). Loss is the AP of the true-task expert minus that of the routed expert.
<table><tr><td rowspan="2">Router</td><td colspan="2">Standard (Order 1)</td><td colspan="2">Long (Order 4)</td></tr><tr><td>Misrouted</td><td>Loss</td><td>Misrouted</td><td>Loss</td></tr><tr><td>Cosine prototype</td><td>2.03%</td><td>1.43</td><td>10.07%</td><td>4.74</td></tr><tr><td>LDA</td><td>0.15%</td><td>0.10</td><td>1.24%</td><td>0.20</td></tr></table>

The cosine router fails by confusing tasks of the same family, because it compares only the direction of the mean embedding and ignores the within-task variance. On Order 4 the largest confusions are MNLI→RTE (22.7% of MNLI inputs), MNLI→CB (13.0%), Yahoo→QQP (6.9%), Yelp→Amazon (5.1%) and Amazon→IMDB (4.0%). LDA whitens with the shared within-task covariance and removes most of these errors. Figure 8 varies the number of support samples per task: cosine routing with $N = 1 0$ is still below its plateau and does not close the gap to LDA on the Long benchmark, and mean-centering the embeddings helps on the Standard benchmark but hurts on the Long benchmark. On the Long benchmark LDA already reaches 94.2% routing accuracy with � = 10, while the cosine router with $N = 1 0 0$ reaches 93.3%; LDA saturates beyond about 500 samples per task.

![](images/6a5d44642eb9a20194d623c53694e67b4e65f0e278bac08205e0684b4ca3a470.jpg)  
Figure 8. Routing accuracy versus support samples per task � (T5-large).

## A.6 Co/De Synergy: Attempts and Limits

We summarise what we tried to make the consolidation branch (Co) improve on the routed expert (De). All numbers are T5-large, three seeds, AP on the Standard (Order 1) and Long (Order 4) benchmarks, reported as Std / Long; rules are selected on a development subset and reported on test.

The two branches are complementary, but no selection rule exploits it. Taking the better of Co and the routed expert per example (the oracle) exceeds the routed expert by 4.7 (Std) to 9.5 (Long) points (Table 9, Figure 9). The oracle is a diagnostic of how many examples Co answers correctly while the expert fails, not an attainable bound. Yet no rule chosen on development data captures it. With the LDA router the posterior is almost always above any useful threshold, so Co answers at most 0.3% of the inputs and CoDe-LoRA reduces to De-LoRA. With the cosine router the fallback does fire, but the routing itself first loses 4.7 points on the Long benchmark, more than Co recovers. The thresholds chosen on development data degenerate to “never use Co”.

![](images/5fff4128928e366a0fb25f378df07af2de86e8d9861e056c4480f7a2797fe615.jpg)  
Figure 9. Per-task accuracy of Co, the routed expert and their oracle on the Long benchmark (Order 4).

Inference-time schemes. We replayed twelve rounds of schemes on the same checkpoints, the same development subset and the same selection protocol (Table 8). Confidence-threshold fallbacks, per-expert gating, output selectors and a 1480- feature selector produced no development-qualified rule or gains below one point that did not hold across protocols. Fusing the two branches on candidate-label scores with a per-task weight was the only scheme that was positive across backbones, and it is the one used in the main tables. Without calibration, summing the two branches hurts on the Long benchmark (−1.0 to −1.6) because the label prior of Co is not removed.

Table 8. Inference-time schemes for combining Co with the routed expert.
<table><tr><td>Scheme family</td><td>Outcome</td></tr><tr><td>Confidence-threshold fallback</td><td>best threshold on development is “never use  ${ \boldsymbol { \mathrm { C o } } } ^ { \flat }$ </td></tr><tr><td>Per-expert branch choice</td><td>≈ routed expert</td></tr><tr><td>Co as an extra expert</td><td>no development-qualified can- didate</td></tr><tr><td>gin, entropy, agree-</td><td>Output selectors (mar- below calibrated De, or kept De</td></tr><tr><td>ment)</td><td>1480-feature selectors none passed development; 26 rescues vs. 26 breaks on test</td></tr><tr><td>tions (PoE, averaging, lem</td><td>Closed-form combina- do not solve the selection prob-</td></tr><tr><td>Dawid-Skene) fusion weight)</td><td>Calibrated label-score positive across backbones; (per-task used in the main tables</td></tr></table>

Decoding-time mixing. Mixing the logprobabilities of Co and the routed expert at every decoding step, $( 1 - w ) \log p _ { \mathrm { D e } } + w \log p _ { \mathrm { C o } } ,$ does not help either. The weight chosen on development data is $w = 0$ on both benchmarks; a per-task weight gives +0.13 (Std) and +0.44 (Long), 2.8% and 5.2% of the oracle gap, at 2.0–2.25× the decoding cost. The branches agree on the top-1 token in 95% (Std) and 91% (Long) of the steps, with log-probability correlations of 0.94 and 0.92, so a mixture with $ w \ = \ 0 . 3$ changes the argmax in only 1.4–1.6% of the steps, and the gains and losses cancel. The complementarity exists at the example level but not at the token level.

Constructions of Co. Table 9 changes only how Co is merged. Projecting against the historical factor space, on one or both sides, or distilling after the SVD, does not improve Co. Merging experts that were trained in isolation from the base model into Co fails, because that shared block is never trained and the generation format collapses (41.87 / 11.06). The null-space merge of Section 4.2 gives a weaker Co that is the most complementary to the experts (oracle 84.53 / 86.00).

Table 9. Constructions of Co (generative, test): Co alone and the oracle that takes the better of Co and the routed expert per example. The routed expert alone scores 79.13 / 76.3–76.5 (Std / Long).
<table><tr><td>Construction</td><td>Co alone</td><td>Oracle</td></tr><tr><td>Trained, SVD merge (main)</td><td>76.67 / 66.33</td><td>83.79 / 84.61</td></tr><tr><td>+ null space, paper scaling</td><td>75.61 / 63.78</td><td>84.53 / 86.00</td></tr><tr><td>+ distillation after SVD</td><td>74.77 / 66.34</td><td>83.66 / 84.51</td></tr><tr><td>+ left-space projection</td><td>76.16 / 65.67</td><td>83.74 / 84.08</td></tr><tr><td>+ two-sided projection</td><td>74.88 / 63.91</td><td>83.14 / 84.60</td></tr><tr><td>Isolated experts, SVD merge</td><td>41.87 / 11.06</td><td>81.84 / 73.99</td></tr><tr><td>+ left projection</td><td>51.31 / 21.96</td><td></td></tr><tr><td>+ two-sided projection</td><td>58.03 / 46.96</td><td></td></tr></table>

Using Co at training time. Initialising each new expert from Co, either by transferring the update or its input subspace, did not pass the development gate across both benchmarks. A family-level initialisation gives +2.30 ± 0.31 on the Long benchmark (positive in 18 of 18 runs over two independent seed sets), but 93% of it comes from the single task CB (+34.1); without CB and COPA it falls to +0.08, and it is no better than a global Co initialisation (+0.03). We therefore report it as a local efect.

A measurement pitfall: routing compensation. With the cosine router (89.9% accurate), fusing paper-style Co with the experts improves over the same calibrated single-branch baseline by +0.49 (Std, 3/3 seeds) and +1.60 (Long, 3/3). Replacing the router by LDA (98.8% accurate) with Co unchanged reduces these to +0.13 (Std, 2/3) and +0.77 (Long, 3/3): about half of the apparent gain was the fusion repairing routing errors, not new information from Co, and 94% of the rest comes from CB and COPA (26 and 40 development examples). When evaluating a two-branch fusion, the routing quality must be fixed and a calibrated single-branch baseline must be reported.

## A.7 Detailed Eficiency Statistics

Table 10 gives the training and inference seconds of every task on the Standard benchmark (Order 1, T5-large, one H20 GPU, seed 41). Inference is the evaluation stage after the task, and the last stage scores all four tasks. Stored adapter parameters are 9.4M (1.28%) for O-LoRA, N-LoRA and CoDe-

LoRA and 32.4M (4.40%) for MoLE-CIE; peak GPU memory is 42.6, 42.6, 31.7 and 41.3 GiB for O-LoRA, N-LoRA, MoLE-CIE and CoDe-LoRA.

Table 10. Training and inference seconds per task on the Standard benchmark (Order 1, T5-large, one H20 GPU, seed 41).
<table><tr><td>Method</td><td>Task</td><td>Train (s) Infer (s)</td><td></td></tr><tr><td rowspan="4">O-LoRA</td><td>DBpedia</td><td>141</td><td>25</td></tr><tr><td>Amazon</td><td>65</td><td>27</td></tr><tr><td>Yahoo</td><td>186</td><td>43</td></tr><tr><td>AG News</td><td>44</td><td>117</td></tr><tr><td rowspan="4">N-LoRA</td><td>DBpedia</td><td>128</td><td>26</td></tr><tr><td>Amazon</td><td>60</td><td>27</td></tr><tr><td>Yahoo</td><td>172</td><td>42</td></tr><tr><td>AG News</td><td>36</td><td>116</td></tr><tr><td rowspan="4">MoLE-CIE</td><td>DBpedia</td><td>767</td><td>128</td></tr><tr><td>Amazon</td><td>626</td><td>110</td></tr><tr><td>Yahoo</td><td>1484</td><td>201</td></tr><tr><td>AG News</td><td>587</td><td>566</td></tr><tr><td rowspan="4">CoDe-LoRA</td><td>DBpedia</td><td>124</td><td>39</td></tr><tr><td>Amazon</td><td>60</td><td>50</td></tr><tr><td>Yahoo</td><td>171</td><td>84</td></tr><tr><td>AG News</td><td>38</td><td>128</td></tr></table>

Long Sequence Eficiency (T=15). Table 11 reports the total training and inference minutes and the stored adapter parameters after the fifteen tasks of the Long Sequence Benchmark (Order 4, T5- large, one H20 GPU). Because only the experts are

Table 11. End-to-end eficiency on the Long Sequence Benchmark (T=15 tasks). CoDe-LoRA is a single seed, the others are means over three seeds.
<table><tr><td>Method</td><td>Train (min) Infer (min)</td><td></td><td>Params</td></tr><tr><td>O-LoRA</td><td>13.3</td><td>9.4</td><td>35.4M (4.80%)</td></tr><tr><td>N-LoRA</td><td>12.0</td><td>9.4</td><td>35.4M (4.80%)</td></tr><tr><td>MoLE-CIE</td><td>140.3</td><td>49.5</td><td>60.0M (8.14%)</td></tr><tr><td>CoDe-LoRA</td><td>12.5</td><td>13.4</td><td>35.4M (4.80%)</td></tr></table>

stored, the parameter growth of CoDe-LoRA over the fifteen tasks is the same as for O-LoRA and N-LoRA.

## A.8 Datasets

In our experiments, we utilize 15 datasets from standard classification and NLU benchmarks (as detailed in Table 12), as well as the generationheavy TRACE benchmark (Wang et al., 2023b). Specifically, the classification datasets are sourced from well-established suites such as the CL benchmark (Zhang et al., 2015), GLUE (Wang et al., 2018), and SuperGLUE (Wang et al., 2019), as well as the IMDB movie review dataset (Maas et al., 2011). These cover natural language inference (NLI), sentiment classification (SC), and topic classification (TC).

Table 12. Datasets and their tasks, categories, domains, and metrics.
<table><tr><td>No.</td><td>Dataset name</td><td>Category</td><td>Task</td><td>Domain</td><td>Metric</td></tr><tr><td>1</td><td>Yelp</td><td>CL Benchmark</td><td>sentiment analysis</td><td>Yelp reviews</td><td>accuracy</td></tr><tr><td>2</td><td>Amazon</td><td>CL Benchmark</td><td>sentiment analysis</td><td>Amazon reviews</td><td>accuracy</td></tr><tr><td>3</td><td>DBpedia</td><td>CL Benchmark</td><td>topic classification</td><td>Wikipedia</td><td>accuracy</td></tr><tr><td>4</td><td>Yahoo</td><td>CL Benchmark</td><td>topic classification</td><td>Yahoo Q&amp;A</td><td>accuracy</td></tr><tr><td>5</td><td>AG News</td><td>CL Benchmark</td><td>topic classification</td><td>news</td><td>accuracy</td></tr><tr><td>6</td><td>MNLI</td><td>GLUE</td><td>NLI</td><td>various</td><td>accuracy</td></tr><tr><td>7</td><td>QQP</td><td>GLUE</td><td>paragraph detection</td><td>Quora</td><td>accuracy</td></tr><tr><td>8</td><td>RTE</td><td>GLUE</td><td>NLI</td><td>news, Wikipedia</td><td>accuracy</td></tr><tr><td>9</td><td>SST-2</td><td>GLUE</td><td>sentiment analysis</td><td>movie reviews</td><td>accuracy</td></tr><tr><td>10</td><td>WiC</td><td>SuperGLUE</td><td>word sense disambiguation</td><td>lexical databases</td><td>accuracy</td></tr><tr><td>11</td><td>CB</td><td>SuperGLUE</td><td>NLI</td><td>various</td><td>accuracy</td></tr><tr><td>12</td><td>COPA</td><td>SuperGLUE</td><td>QA</td><td>blogs, encyclopedia</td><td>accuracy</td></tr><tr><td>13</td><td>BoolQA</td><td>SuperGLUE</td><td>boolean QA</td><td>Wikipedia</td><td>accuracy</td></tr><tr><td>14</td><td>MultiRC</td><td>SuperGLUE</td><td>QA</td><td>various</td><td>accuracy</td></tr><tr><td>15</td><td>IMDB</td><td>SuperGLUE</td><td>sentiment analysis</td><td>movie reviews</td><td>accuracy</td></tr></table>

Crucially, to evaluate generalization on more complex, generation-heavy tasks, we also incorporate the TRACE benchmark (Wang et al., 2023b). Unlike standard classification tasks, TRACE features multi-turn, complex generative tasks that assess the structural reasoning and domain adaptation of LLMs. In particular, we evaluate on code generation (completion) using the Py150 dataset, where even a single syntax error can lead to a compilation failure, making it highly fragile under sequential fine-tuning.

## A.9 Task Sequence and Instructions

To evaluate the robustness of our method, we employ a diverse set of task instructions and evaluate across multiple task orders. Table 13 summarizes the specific natural language prompts used for each task category, following the instruction-tuning format described in Qin and Joty (2021). The task sequences for the T5 and Llama 2-7B experiments, together with the TRACE benchmark sequence, are provided in Table 14.

Table 13. Task descriptions and corresponding prompts.
<table><tr><td>Task</td><td>Prompts</td></tr><tr><td>NLI</td><td>What is the logical relationship between the &quot;sentence 1” and the &quot;sentence 2&quot;? Choose one from the option.</td></tr><tr><td>QQP</td><td>Whether the “first sentence&quot;and the “second sentence&quot; have the same meaning? Choose one from the option.</td></tr><tr><td>SC</td><td>What is the sentiment of the following paragraph? Choose one from the option.</td></tr><tr><td>TC</td><td>What is the topic of the following paragraph? Choose one from the option.</td></tr><tr><td>BoolQA</td><td>According to the following passage, is the question true or false? Choose one from the option.</td></tr><tr><td>MultiRC</td><td>According to the following passage and question, is the candidate answer true or false? Choose one from the option.</td></tr><tr><td>WiC</td><td>Given a word and two sentences, whether the word is used with the same sense in both sentences? Choose one from the option.</td></tr></table>

Table 14. Task sequences for diferent model and benchmark settings.
<table><tr><td></td><td>Order Setting</td><td>Task Sequence</td></tr><tr><td>1</td><td></td><td>T5, Llama 2-7B dbpedia → amazon → yahoo → ag</td></tr><tr><td>2</td><td></td><td>T5, Llama 2-7B dbpedia → amazon → ag → yahoo</td></tr><tr><td>3</td><td></td><td>T5, Llama 2-7B yahoo → amazon → ag → dbpedia</td></tr><tr><td>4</td><td>T5</td><td> $\mathrm { \ m n l i \to c b \to w i c \to c o p a \to q q p \to b o o l q a \to }$  rte → imdb → yelp → amazon → sst-2 → dbpedia → ag → multirc → yahoo</td></tr><tr><td>5</td><td>T5</td><td>multirc → boolqa → wic → mnli → cb → copa → qqp → rte → imdb → sst-2 → dbpedia → ag</td></tr><tr><td>6</td><td>T5</td><td>→ yelp → amazon → yahoo yelp → amazon → mnli → cb → copa → qqp → rte → imdb → sst-2 → dbpedia → ag → yahoo → multirc → boolqa → wic</td></tr><tr><td>1</td><td>TRACE</td><td>C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds</td></tr></table>