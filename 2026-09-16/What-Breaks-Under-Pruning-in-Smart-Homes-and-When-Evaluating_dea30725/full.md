# What Breaks Under Pruning in Smart Homes, and When? Evaluating LLM Degradation Across Architectures and Task Complexity

Congjing Zhang   
Alexa Home AI, Amazon.com   
University of Washington   
cjzhang@amazon.com   
congjing@uw.edu

Henning Lange Alexa Home AI, Amazon.com helange@amazon.com

Vashishtha Patil<sup>†</sup> Alexa Home AI, Amazon.com pativash@amazon.com

Usman Aleem Alexa Home AI, Amazon.com ualeem@amazon.com

## Abstract

Pruning can reduce the deployment cost of large language models (LLMs), but its impact on context-grounded tool calling remains poorly understood. We systematically study pruning-induced degradation in smart-home tool calling across four LLMs spanning dense Transformer, dense hybrid, and mixture-ofexperts (MoE) architectures, together with depth, width, hybrid, and expert pruning methods. After post-pruning supervised fine-tuning (SFT), we evaluate more than 19,500 instances from three smart-home datasets. Beyond aggregate task accuracy, we characterize degradation along two dimensions: action components (i.e., operation, device, argument, and value) and task complexity. Our results show that dense models have narrow safe pruning regions followed by sharp degradation, while MoE models tolerate substantially more pruning. Pruning degrades grounded specificity before schemalevel intent, and aggressive dense pruning can induce systematic over-refusal. These findings highlight the importance of evaluating pruning beyond aggregate accuracy when selecting pruned LLMs for reliable tool execution.

## 1 Introduction

Large language models (LLMs) are increasingly used as natural-language interfaces for smart-home control, translating user requests into structured actions grounded in household devices, capabilities, and states (Rivkin et al., 2024; Zhang et al., 2026b). However, deploying LLMs can incur substantial memory and computational costs. Structured pruning reduces these costs by removing components such as Transformer layers, channels, or experts while preserving hardware-friendly model structures (Ma et al., 2023; Gao et al., 2024). For deployment, an important question is therefore not only how much an LLM can be pruned, but what capabilities are lost as capacity is removed.

This question is particularly important for smarthome tool execution, where a single action requires multiple decisions: what operation to perform, which device to target, which argument to control, and what value to use (Seo et al., 2026). Requests also vary in complexity, from singledevice operations to multi-action, contextual, partially executable, and infeasible requests (Li et al., 2025). Aggregate accuracy can therefore hide substantially different pruning-induced failure modes. Prior work shows that pruning can disproportionately impair difficult downstream tasks (Yin et al., 2024), but existing studies largely evaluate pruned LLMs using aggregate benchmarks, while smarthome benchmarks primarily study unpruned models. It remains unclear how pruning sensitivity varies across architectures, action components, and task complexity, or whether aggressive pruning can qualitatively change model behavior rather than simply reduce accuracy.

We systematically study these questions across four LLMs spanning dense Transformer, dense hybrid, and sparse mixture-of-experts (MoE) architectures, using representative depth, width, hybrid, and expert-pruning methods. Each pruned model undergoes supervised fine-tuning (SFT), followed by evaluation on more than 19,500 instances from three smart-home datasets. Beyond aggregate task accuracy, we analyze degradation along two dimensions: action components (operation, device, argument, and value) and task complexity (Simple, Medium, Complex, Partially Executable, and Infeasible). We find that dense models exhibit narrow safe pruning regions followed by sharp degradation, whereas the MoE model tolerates greater expert removal. Pruning also degrades grounded specificity (i.e., device and value prediction) before schemalevel intent, while baseline task difficulty does not reliably show pruning sensitivity. Under aggressive dense pruning, models can further shift toward systematic over-refusal of executable requests.

To sum up, our main contributions are: (1) a fine-grained framework for analyzing pruninginduced degradation across action components and smart-home task-complexity levels; (2) a systematic evaluation across three architecture families, four classes of structured pruning, and three smarthome datasets with post-pruning SFT; and (3) empirical characterization of deployment-relevant failure patterns, including pruning cliffs, specificityfirst degradation, task-dependent pruning sensitivity, and over-refusal under aggressive pruning.

## 2 Related Work

LLM pruning. Structured pruning improves LLM inference efficiency by removing redundant structural components, such as channels, Transformer layers, or experts (Ma et al., 2023; Gao et al., 2024). Existing methods operate at different structural levels: FLAP (An et al., 2024) and Slim-LLM (Guo et al., 2025) prune width-level components such as channels and attention heads, while ShortGPT (Men et al., 2025) and Gromov et al. (2025) identify and remove redundant Transformer layers. Hybrid approaches such as 2SSP (Sandri et al., 2025) combine depth and width pruning, and recent methods extend structured pruning to globally optimized settings and MoE models (Lasby et al., 2026; Zhang et al., 2026a). However, most pruning studies evaluate aggregate accuracy, which can obscure capability-specific degradation.

LLMs for smart homes. LLMs have increasingly been explored as natural-language interfaces and autonomous agents for smart-home control (Chen et al., 2026). Rivkin et al. (2024) develop an LLM-based agent that reasons over household context and interacts with smart-home devices. HomeBench (Li et al., 2025) studies valid and invalid instructions involving single and multiple devices. SimuHome (Seo et al., 2026) introduces temporal and environment-aware interactions. SMH-Bench (Li et al., 2026) further evaluates environment-grounded reasoning and action across complex smart-home scenarios. These works primarily study unpruned LLMs for smart homes. The reliability of pruned LLMs for smarthome tool execution remains largely unexplored.

## 3 Evaluation Framework

Figure 1 summarizes our evaluation framework. We apply various pruning methods to LLMs with different architectural designs and recover the pruned models through SFT. We then evaluate reliability at two complementary granularities. At the action level, we decompose each generated smart-home operation into its device, operation, argument, and value components to identify which part of structured execution is most affected by pruning. At the task level, we stratify requests by smart-home complexity to characterize when pruning-induced degradation emerges. Together, these two views allow us to measure not only how much reliability is lost under pruning, but also what breaks and under which task conditions.

## 3.1 Problem Formulation

Following Seo et al. (2026); Li et al. (2026), we formulate smart-home tool calling as contextgrounded structured prediction. Each instance is represented by $X = \{ { \pmb u } , H , A \}$ , where u is the user request, H is the smart-home context, and A is the set of available device actions. The context H specifies devices, their capabilities, and current device or environmental states. Given X, an LLM produces a set of executions by $Y =$ $\{ { \pmb a } _ { 1 } , { \pmb a } _ { 2 } , \dots , { \pmb a } _ { T } \}$ , where $T$ is the number of actions a required to fulfill the user request u. We define each action $\mathbf { \delta } _ { a _ { j } , j } = 1 , \ldots , T$ into four semantic components ${ \pmb a } _ { j } = ( o _ { j } , d _ { j } , p _ { j } , v _ { j } )$ , where $o _ { j }$ is the operation to be performed, $d _ { j }$ is the targetdevice, $p _ { j }$ is the control-argument names, and $v _ { j }$ is the argument value. For example, for the request “dim the bedroom light to $5 0 ^ { \circ }$ , consider the action “bedroom.light.set.brightness(50)”. It is represented as $d = ^ { 6 6 }$ “bedroom.light”, $o = \tilde { \mathbf { \Gamma } } \mathbf { s e t } ^ { \prime \prime } , p =$ “brightness”, and $v = 5 0$

## 3.2 Smart-Home Task-Complexity Taxonomy

To characterize pruning sensitivity across request types, we partition evaluation instances into five mutually exclusive categories based on execution complexity and feasibility.

Simple. A single-turn request requiring one action on one target device, e.g., “turn off the kitchen

![](images/df67220e6818e997abf5ebec464ee1eb2c418614e3a778d118630c8a0f9cceb9.jpg)  
Figure 1: Overview of our framework for evaluating pruning-induced reliability degradation in smart-home tool execution across action components and task-complexity levels.

light.”

Medium. A single-turn request requiring multiple actions or multiple target devices, or an underspecified request that requires clarification but not prior dialogue context.

Complex. A request whose correct execution depends on prior interaction context, such as resolving anaphoric references or repeating, reversing, or modifying a previous action.

Partially Executable. Only a subset of the requested actions is feasible; the model must execute the valid portion while rejecting the invalid portion.

Infeasible. None of the requested actions is feasible under the current smart-home context, so the model should avoid execution and reject X.

## 3.3 LLM Architecture Families

To examine whether pruning-induced degradation depends on model architecture, we evaluate four models from the Qwen family spanning dense Transformer, dense hybrid, and MoE designs. Qwen3-4B (Team, 2025) is a 4B-parameter dense Transformer with 36 layers and grouped-query attention (GQA). Qwen3.5-4B and Qwen3.5-9B (Qwen Team, 2026a) adopt a dense hybrid architecture that interleaves Gated DeltaNet linear-attention layers with full-attention layers; both contain 32 layers. Finally, Qwen3.6-35B-A3B (Qwen Team, 2026b) extends this hybrid design with sparse MoE feed-forward layers, comprising 35B total parameters while activating around 3B parameters per token across 40 layers.

This selection enables us to study three complementary factors. Comparing Qwen3-4B with Qwen3.5-4B isolates changes in architectural design at approximately fixed parameter scale; comparing Qwen3.5-4B with Qwen3.5-9B examines the effect of dense model capacity; and Qwen3.6- 35B-A3B provides a sparse MoE architecture whose parameter allocation differs fundamentally from dense models. Together, these models allow us to test whether pruning affects smart-home reliability consistently across architectural families or produces architecture-specific failure patterns.

## 3.4 Pruning Methods

We evaluate pruning across four granularities: depth, width, hybrid, and expert pruning, subject to architectural compatibility.

Depth pruning. We use ShortGPT (Men et al., 2025), which removes low-importance layers based on Block Influence, and Angular (Gromov et al., 2025), which identifies redundant contiguous layer blocks using angular similarity of boundary representations. Both are applied to Qwen3-4B, Qwen3.5-4B, and Qwen3.5-9B.

Width pruning. We use FLAP (An et al., 2024), which prunes feature channels using a fluctuationbased importance criterion with adaptive sparsity allocation and bias compensation. We apply FLAP to Qwen3-4B; its method design does not directly support the Gated DeltaNet modules used in Qwen3.5.

Hybrid pruning. We use 2SSP (Sandri et al., 2025), which combines feed-forward neuron pruning with attention-submodule pruning. We apply 2SSP to Qwen3-4B; its attention-pruning stage assumes conventional attention modules and would require architecture-specific modifications for Qwen3.5.

Expert pruning. For Qwen3.6-35B-A3B, we use REAP (Lasby et al., 2026), which ranks experts using router scores and activation magnitudes and removes those with low estimated contribution.

After pruning, all models undergo the same SFT procedure while keeping the pruned architecture fixed. Overall, Qwen3-4B is evaluated with Short-GPT, Angular, FLAP, and 2SSP; Qwen3.5-4B and Qwen3.5-9B with ShortGPT and Angular; and Qwen3.6-35B-A3B with REAP.

## 3.5 Pruning-Degradation Evaluation

Let $M _ { 0 }$ denote the original SFT LLM without pruning and $M _ { \pi }$ an LLM obtained under pruning configuration π, which specifies the pruning method and ratio. We evaluate performance at two complementary granularities: action-component level, which identifies what parts of structured tool execution are affected by pruning, and taskcomplexity level, which identifies under what task conditions degradation occurs. For evaluation instance $X _ { i } , i = 1 , \ldots , N$ , let $Y _ { i }$ denote the groundtruth action set and $M ( X _ { i } )$ the action set produced by model M. Overall task accuracy is $\begin{array} { r } { \operatorname { A c c } ( M ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ M ( X _ { i } ) = Y _ { i } ] } \end{array}$ . An instance is considered correct only when the complete required execution of a is satisfied. For multi-action requests, actions are matched independently of their order.

Action-component accuracy. We evaluate the four action components $\{ o , d , p , v \}$ defined in Section 3.1. For instance $X _ { i }$ , let $Y _ { i } ^ { ( j ) }$ and $M ^ { ( j ) } ( X _ { i } )$ denote its ground-truth and predicted component from model M, where $j ~ \in ~ \{ o , d , p , v \}$ , respectively. Component $j ^ { \prime } { \bf s }$ accuracy of M is $\begin{array} { r } { \operatorname { A c c } ^ { ( j ) } ( M ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \Big [ M ^ { ( j ) } ( X _ { i } ) = Y _ { i } ^ { ( j ) } \Big ] } \end{array}$

Complexity-level accuracy. We report task accuracy separately for each category $c \in$ {Simple, Medium, Complex, Partially Executable, Infeasible} in the complexity taxonomy of Section 3.2. Let $I _ { c }$ denote the set of evaluation instances belonging to category c. We define the category c’s accuracy of M as $\operatorname { A c c } _ { c } ( M ) =$ $\begin{array} { r } { \frac { 1 } { | I _ { c } | } \sum _ { i \in I _ { c } } \mathbb { I } [ M ( X _ { i } ) = Y _ { i } ] } \end{array}$

<table><tr><td>Complexity Category</td><td>SHTC (%)</td><td>HomeBench</td><td>HAR</td></tr><tr><td>Simple</td><td>66.6%</td><td>6,187</td><td>1,903</td></tr><tr><td>Medium</td><td>29.4%</td><td>245</td><td>291</td></tr><tr><td>Complex</td><td>3.9%</td><td></td><td></td></tr><tr><td>Partially Executable</td><td>一</td><td>4,072</td><td>一</td></tr><tr><td>Infeasible</td><td></td><td>6,862</td><td></td></tr><tr><td>Total</td><td>100%</td><td>17,366</td><td>2,194</td></tr></table>

Table 1: Statistics of the evaluation sets. For SHTC, we report the proportion of instances in each category.

Pruning-induced degradation. We measure reliability degradation relative to the corresponding unpruned model: $\Delta \mathrm { A c c } ( M _ { \pi } )$ = $\begin{array} { r l r } { \operatorname { A c c } ( M _ { \pi } ) } & { { } - } & { \operatorname { A c c } ( M _ { 0 } ) , \operatorname { \Delta A c c } ^ { ( k ) } ( M _ { \pi } ) } \end{array}$ = $\begin{array} { r l r } { \operatorname { A c c } ^ { ( k ) } ( M _ { \pi } ) } & { { } - } & { \operatorname { A c c } ^ { ( k ) } ( M _ { 0 } ) , \operatorname { \Delta A c c } _ { c } ( M _ { \pi } ) } \end{array} \quad =$ $\operatorname { A c c } _ { c } ( M _ { \pi } ) - \operatorname { A c c } _ { c } ( M _ { 0 } )$ . We report these changes with negative values indicating degradation after pruning. Action-component changes identify which parts of structured tool execution are most sensitive to capacity reduction, while complexitylevel changes identify the task conditions under which these failures are most pronounced.

## 4 Experiments

## 4.1 Experimental Setup

For SFT and evaluation datasets, we use the training and test splits, respectively, from the same three sources: proprietary smart-home tool-calling dialogues (SHTC), HomeBench (Li et al., 2025), and Home-Assistant-Requests-V2 (HAR)<sup>1</sup>. All pruned models are healed using the same SFT mixture constructed from the training splits. Table 1 shows the statistics of the evaluation set. For SHTC, to comply with proprietary-data requirements, we report only performance changes ∆Acc relative to the corresponding unpruned model $M _ { 0 }$ . Additional details of experimental setup are provided in Appendix A.

## 4.2 Results

We show HomeBench action-component and complexity-level accuracy for Qwen3-4B in Figures 2 and 3. Complete results are provided in Appendix Figures 9-11 and 12-14. Here, we focus on the key findings and insights.

Dense pruning exhibits a narrower safe region than MoE pruning. Figure 4 contrasts the dense and MoE pruning curves. Overall, dense pruning exhibits sharp cliffs, whereas MoE pruning maintains a wider plateau. At 10% dense pruning, seven of eight LLM-method combinations lose less than 0.6% accuracy, suggesting that mild pruning mainly removes redundant capacity. Pruning ratio alone does not determine retained capability; which structures are removed is also critical. The dense hybrid Qwen3.5-4B is more robust than the same-scale GQA model, possibly because its heterogeneous Gated DeltaNet and full-attention layers provide more alternative information pathways. However, increasing the hybrid model from 4B to 9B yields no consistent benefit, suggesting that additional parameters do not necessarily correspond to pruning-redundant capacity. For REAP, removing 70% of experts reduces stored parameters from 35B to approximately 12.5B with a nonsignificant −0.37% change. Unlike dense pruning, REAP reduces stored experts while preserving routing and eight active experts per token (around 3B parameters), which explains its wider safe region.

![](images/12a3f1e8a7bde24da37768952b884063368dfb3fbb49858dab30ac5fbc9461bf.jpg)  
(a) ShortGPT

![](images/72068bded0ce1fe7a6fab0fba03bf9e50d4db881ace05fc9444166436140a6f9.jpg)  
(b) Angular

![](images/be4936b929e0b0b3db02a69a2d72140d98b4b053ac3798538cedb0a5b98a36a9.jpg)  
(c) FLAP

![](images/521c1926db0658ed462f39d6a33eaff139f78995e5e6ff85baafcb0f35add4e4.jpg)  
(d) 2SSP

Figure 2: HomeBench action-component accuracy across pruning ratios for four pruning methods in Qwen3-4B.  
![](images/0cf8a629c026534ff434218f9bfaf4a3e29965a86a09f7db004d5df8319ee9ed.jpg)  
(a) ShortGPT

![](images/26c1e070f0541a3e8512a49e8a7221a8cb53ed3abb7dd9b82fffc890e045af26.jpg)  
(b) Angular

![](images/12d6cce674133e550b65ad3998f88e41ab78c9b99c4d000711e2171b35f63faa.jpg)  
(c) FLAP

![](images/1b1a2c80017b1eaeaab3d91dc4b24966cafa0c04b8bcace12526949ece5fb74b.jpg)  
(d) 2SSP  
Figure 3: HomeBench complexity-level accuracy across pruning ratios for four pruning methods in Qwen3-4B.

Pruning removes specificity before intent. We distinguish schema-level intent, represented by operation and argument $( o , p )$ , from grounded specificity, represented by device and value (d, v). We compare device with operation because o specifies what to do and d where to do it; similarly, p identifies the controlled attribute and v its requested setting. Figure 5 plots $\Delta \operatorname { A c c } ^ { ( d ) } - \Delta \operatorname { A c c } ^ { \mathsf { ( \alpha ) } }$ and $\Delta \operatorname { A c c } ^ { ( v ) } - \Delta \operatorname { A c c } ^ { ( p ) }$ , averaged across dense pruning results over all three datasets; negative values indicate greater degradation of specificity than intent. Both gaps become increasingly negative with pruning. At 50%, mean losses are 24.2% vs. 21.1% for d vs. o, and 20.7% vs. 14.5% for v vs. p, with both specificity components degrading at least as much as their paired intent components. It is because o and p come from a relatively small, repeated schema, whereas d and v require instance-specific grounding in the provided context and precise selection among candidates. d is also the weakest unpruned component, suggesting that pruning amplifies an already difficult grounding step. Thus, reduced capacity appears to preserve the general action schema longer than the contextual details needed to execute it precisely.

Baseline accuracy does not predict pruning sensitivity. Figure 6 shows that pruning sensitivity is not monotonic in unpruned accuracy. Partially Executable requests are the most sensitive, losing about 10.3% accuracy per additional 10% pruning, followed by Simple (7.7%) and Medium (7.2%) requests. Notably, Simple and Infeasible requests have similarly high unpruned accuracy, yet Simple requests degrade faster than Infeasible requests. Conversely, Medium requests begin at much lower accuracy but exhibit pruning sensitivity comparable to Simple requests. These contrasts show that baseline difficulty alone does not determine robustness to capacity reduction. Partially Executable requests are especially vulnerable because they require jointly identifying feasible and infeasible portions while coordinating execution and rejection. Simple requests, despite being easy for the unpruned model, still require precise grounding to a device and action, which can deteriorate under pruning. Infeasible requests instead primarily require withholding action; together with Figure 7, their relative robustness may partly reflect pruninginduced preference for rejection.

![](images/86fc8f7ab7e1b4bd74ba9c36293dbd90e379a0d0df40dcfe253dbf8f81d9463a.jpg)  
Figure 4: Average ∆Acc of different LLMs under different pruning methods across three datasets.

![](images/e3ae3b18a60c678ab7a8d59a61b67fcd830aac5ca7d17f0d0c3d85160f520495.jpg)  
Figure 5: Specificity gaps $\Delta \mathrm { A c c } ^ { ( d ) }$ $\Delta \ * { \check { \mathrm { A c c } } } ^ { ( o ) }$ and $\Delta \mathrm { A c c } ^ { ( v ) } ~ - ~ \Delta \mathrm { A c c } ^ { ( p ) }$ across dense pruning configurations.

![](images/68f240b70a4b59eddfb4390694076debe4d42d9f2290019761c3f80a61cbf470.jpg)  
Figure 6: $| \Delta \mathrm { A c c } _ { c } |$ per additional 10% pruning versus mean $\operatorname { A c c } _ { c } ( M _ { 0 } )$ across public datasets.

Deep pruning can collapse into over-refusal. Figure 7 shows that aggressive pruning increasingly shifts the model from incorrect

![](images/30d1d592c720c93dad156b04e54f90ec72c71a752b2c073f7638bd477f47e69e.jpg)  
Figure 7: False-refusal rate on executable HomeBench requests.

execution toward refusing to act using Qwen3-4B. On executable HomeBench requests, unpruned Qwen3-4B refuses 7.7%; at 50% pruning this rises to 56.2% with Angular and 80.8% with FLAP. Yet these models still withhold action on 86.8% and 99.9% of infeasible requests, respectively. One possible explanation is that generating a valid tool call requires precise device, operation, and value grounding, whereas refusal is a simpler low-commitment output. As pruning weakens grounded generation, the model may therefore increasingly default to rejection when uncertain. Consequently, an infeasible-only refusal metric would make the FLAP model appear robust while it rejects four out of five valid commands. Deep pruning thus changes the model’s decision policy, not merely its execution accuracy.

## 5 Discussion and Future Work

Our results show that pruning in smart-home tool calling must preserve more than general semantic capability. Unlike conventional language or structured-prediction tasks, correct execution is conditioned on the current complex homeenvironment. Operation and argument prediction remain stable longer than device and value prediction, indicating that pruning can preserve the general action schema while damaging the contextual specificity. Partially Executable requests are especially sensitive, and aggressive pruning can shift models toward over-refusal. Thus, aggregate accuracy alone is insufficient in smart homes. For bounded-domain MoEs, expert pruning is a promising starting point. For dense LLMs, pruning should be more conservative, with multiple ratios and criteria evaluated for degradation.

In smart homes, future work should develop pruning and recovery methods that explicitly preserve environment grounding. Pruning criteria could protect components important for device resolution and precise value prediction rather than relying only on generic importance scores. Healing data should target pruning-specific failures, including distractor-rich device resolution, precise values, multi-device requests, and partially executable instructions. Because aggressive pruning can induce over-refusal, recovery should also calibrate the execute-versus-reject decision using feasible and infeasible requests. Unlike conventional domains where recovery may mainly restore aggregate language or reasoning performance, smart-home healing must restore the link between language, the current environment, and selective action.

## 6 Conclusion

We systematically study how pruning affects LLMs for context-grounded smart-home tool calling across architectures, pruning methods, and severities. Beyond aggregate accuracy, we examine degradation at the action-component and taskcomplexity levels. We find that grounded specificity degrades before schema-level intent, and aggressive dense pruning can induce over-refusal. These findings show that pruning should be evaluated not only by overall accuracy, but also by architecture, workload, and failure mode. Fine-grained evaluation is therefore important for identifying pruning regimes that reduce model cost while preserving reliable smart-home tool execution.

## Limitations

Despite the insights, our study has several limitations. First, although we cover dense Transformer, dense hybrid, and MoE architectures, all evaluated models belong to the Qwen family; whether the observed pruning patterns generalize to other model families remains to be studied. Second, pruning methods are not evaluated in a fully factorial manner because some methods are architecturespecific; for example, FLAP and 2SSP are not directly applicable to the Gated DeltaNet-based models. Finally, our evaluation is restricted to smarthome tool calling, and some complexity categories are available in only a subset of the three datasets, which limits broader generalization across domains. Future work could extend the analysis to additional model families, domains, pruning methods, and deployment-level efficiency measures such as latency and throughput.

## References

Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. 2024. Fluctuation-based adaptive structured pruning for large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 10865–10873.

Maximillian Chen, Xuanming Zhang, Michael Peng, Zhou Yu, Alexandros Papangelis, and Yohan Jo. 2026. Mist: Multimodal interactive speech-based tool-calling conversational assistants for smart homes. arXiv preprint arXiv:2605.06897.

Shangqian Gao, Chi-Heng Lin, Ting Hua, Tang Zheng, Yilin Shen, Hongxia Jin, and Yen-Chang Hsu. 2024. Disp-llm: Dimension-independent structural pruning for large language models. Advances in Neural Information Processing Systems, 37:72219–72244.

Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A Roberts. 2025. The unreasonable ineffectiveness of the deeper layers. In

International Conference on Learning Representations, volume 2025, pages 81906–81920.

Jialong Guo, Xinghao Chen, Yehui Tang, and Yunhe Wang. 2025. SlimLLM: Accurate structured pruning for large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 20766–20776. PMLR.

Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, and Vithursan Thangarasa. 2026. Reap the experts: Why pruning prevails for oneshot moe compression. In International Conference on Learning Representations, volume 2026, pages 146883–146912.

Kuan Li, Shuo Zhang, Huacan Wang, Fangzhou Yu, Zecheng Sheng, Yi Gu, Weipeng Ming, Lei Xue, Chen Liu, Sen Hu, et al. 2026. Smh-bench: Benchmarking llm agents for environment-grounded reasoning and action in smart homes. arXiv preprint arXiv:2606.01912.

Silin Li, Yuhang Guo, Jiashu Yao, Zeming Liu, and Haifeng Wang. 2025. Homebench: Evaluating llms in smart homes with valid and invalid instructions across single and multiple devices. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12230–12250.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. Llm-pruner: On the structural pruning of large language models. Advances in neural information processing systems, 36:21702–21720.

Xin Men, Mingyu Xu, Qingyu Zhang, Qianhao Yuan, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. 2025. Shortgpt: Layers in large language models are more redundant than you expect. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 20192–20204.

Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.

Qwen Team. 2026b. Qwen3.6-35B-A3B: Agentic coding power, now open to all.

Dmitriy Rivkin, Francois Hogan, Amal Feriani, Abhisek Konar, Adam Sigal, Xue Liu, and Gregory Dudek. 2024. Aiot smart home via autonomous llm agents. IEEE Internet ofThings Journal, 12(3):2458–2472.

Fabrizio Sandri, Elia Cunegatti, and Giovanni Iacca. 2025. 2SSP: A two-stage framework for structured pruning of LLMs. Transactions on Machine Learning Research.

Gyuhyeon Seo, Jungwoo Yang, Junseong Pyo, Nalim Kim, Jonggeun Lee, and Yohan Jo. 2026. Simuhome: A temporal-and environment-aware benchmark for smart home llm agents. In International Conference on Learning Representations, volume 2026, pages 40070–40097.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Lu Yin, Ajay Kumar Jaiswal, Shiwei Liu, Souvik Kundu, and Zhangyang Wang. 2024. Junk DNA hypothesis: Pruning small pre-trained weights Irreversibly and Monotonically impairs “difficult" downstream tasks in LLMs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 57053–57068. PMLR.

Geng Zhang, Han Yuxuan, Yuxuan Lou, Yiqi Zhang, Wangbo Zhao, and Yang You. 2026a. Mone: Replacing redundant experts with lightweight novices for structured pruning of moe. In International Conference on Learning Representations, volume 2026, pages 39983–40009.

Zhengyuan Zhang, Dong Zhao, Tiancheng He, Zilong Wang, Xiangyu Li, and Huadong Ma. 2026b. Vcullm: Prompt-efficient on-device large language model for vague command understanding in smart homes. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 10(2):1–30.

## A Additional Experimental Setup

Pruning Ratios. Dense LLM architectures are pruned at ratios from 10% to 50% in increments of 10%. For the MoE architecture, we evaluate more aggressive expert pruning, removing 10% to 90% of experts to account for its higher structural sparsity.

SFT Data. The SFT data mixture is balanced three ways by source rather than pooled in natural proportion: 16,667 examples each from SHTC, HomeBench and HAR, giving 50,001 training examples, with 100 further examples per source held out for validation.

Training Settings. SFT is performed for one epoch with a global batch size of 32 and a microbatch size of 1, resulting in 1,563 optimizer steps. We use a maximum sequence length of 32,768 tokens and bf16 mixed precision. Optimization uses Adam with $\beta _ { 2 } { = } 0 . 9 8$ and an initial learning rate of $5 \times 1 0 ^ { - 6 }$ , with 50 warmup steps followed by decay to a minimum learning rate of $5 \times 1 0 ^ { - 7 }$ . We use 32 NVIDIA A100 80 GB GPUs for dense LLM architectures and 64 NVIDIA A100 80 GB GPUs for the MoE model.

Generation Settings. Unless stated otherwise, we use greedy decoding with temperature as 0. To quantify the extent to which per-instance variation may arise from sampling stochasticity, we additionally evaluate Qwen-recommended sampling settings with temperature as 0.7, top- $\cdot p = 0 . 8$ , and top-k = 20.

## B Prompt Example

Figure 8 shows an example prompt used for smarthome tool calling in HomeBench. The prompt provides the LLM with the current smart-home state, including available devices, their attributes, and valid value ranges, together with the device-control methods that may be invoked. The prompt also specifies the required machine-instruction output format. The target user request is then appended at the end of the prompt, and the model is required to generate the corresponding executable action(s) using only the provided devices and methods. For unsupported devices or attributes, the prompt instructs the model to return “error\_input”.

![](images/9cb7d60dab15184460cbc22174dea211dec90c13b5c209d9bf30326b704f54b4.jpg)  
Figure 8: Prompt example for HomeBench.

## C Action-Component Accuracy

Figures 9-11 provide the complete actioncomponent results across datasets, LLMs, pruning methods, and pruning ratios. Overall, the results support the pattern discussed in Section 4.2: dense models generally preserve all four components under mild pruning, followed by increasingly component-specific degradation at higher pruning ratios. In particular, device (d) and value (v) accuracy often decline more rapidly than operation (o) and argument (p) accuracy, indicating that pruning tends to damage grounded specificity before schema-level intent. This behavior is consistent across HomeBench (Figure 9), HAR (Figure 10), and SHTC (Figure 11). In contrast, the MoE model pruned with REAP maintains relatively stable component-level performance over a much wider range of pruning ratios, with noticeable degradation appearing primarily under the most aggressive expert pruning.

## D Complexity-Level Accuracy

Figures 12-14 present the complete complexitylevel results across datasets, LLM architectures, pruning methods, and pruning ratios. The results further show that pruning sensitivity is not determined by the unpruned accuracy alone. For HomeBench (Figure 12), Medium and Partially Executable requests generally degrade more rapidly under aggressive dense pruning, while Infeasible requests remain comparatively robust. The HAR results (Figure 13) similarly show substantial degradation of Medium requests for several densepruning configurations, although the magnitude varies across architectures and methods. For SHTC (Figure 14), Simple and Medium requests generally experience larger losses than Complex requests at high dense-pruning ratios, further illustrating that baseline task difficulty does not directly determine pruning sensitivity. Across all three datasets, degradation becomes substantially more categorydependent as dense pruning becomes aggressive. In contrast, the MoE model with REAP preserves relatively stable accuracy across task categories over a much wider pruning range, with degradation emerging primarily at the highest expert-pruning ratios.

## E Operation-Level F1 Scores

Figures 15-20 repeat the action-component and complexity-level analyses of Appendices C and D with F1 scores in place of Acc. The actioncomponent results are Figures 15 (HomeBench), 16 (HAR) and 17 (SHTC), and the complexitylevel results are Figures 18, 19 and 20. We use the operation-level F1 following Li et al. (2025). Precision (P) is the number of operations the model predicts correctly divided by the number of operations it predicts; recall (R) is the number of operations it predicts correctly divided by the number of operations the user instruction actually requires, and $\mathrm { F } 1 = 2 P R / ( P + R )$

F1 does not change the conclusions, and the shape of the degradation is unchanged. Averaged over every LLM-pruning pair, the drop from $M _ { 0 }$ to the most aggressive ratio decomposes almost identically under the two metrics: on SHTC d loses 28.7% of F1 against 20.9% for $p ,$ 21.5% for v and 12.9% for $^ { O , }$ and on HAR d loses 17.6% points against 9.5%, 9.6% and 7.8%. Grounded target selection therefore remains the component that pruning erodes first, dense pruning still shows a narrow safe region followed by a category-dependent collapse, and the MoE model under REAP still stays flat over a far wider pruning range.

## F Robustness Analyses

## F.1 Robustness Across the Pruning Ratios

To study the behavioral robustness across pruning ratios, we use the results at each REAP ratio for Qwen3.6-35B-A3B. For each instance and metric, we classify the outcome across the ten checkpoints from 0% to 90% as always correct, always wrong, or flipping across ratios. Thus, this analysis measures whether the same instances remain correct throughout the pruning sweep, rather than variation across random seeds.

Figure 21 shows that robust aggregate accuracy does not imply stable instance-level behavior. On HomeBench, 14.1% of exact-match outcomes change somewhere along the pruning, rising to 33.7% for Partially Executable and 24.9% for Medium requests, despite the nearly flat aggregate curve. HAR has a larger exact-match flipping band (25.1%), concentrated in device grounding (24.4%) and Simple requests (27.4%); its Medium band is instead mostly always wrong (78.7%). On SHTC, Medium is again the least stable category (29.4% flipping), while operation selection is the most stable action component (11.0%). These results distinguish a stable mean from stable predictions: expert pruning can preserve aggregate accuracy while changing which requests succeed.

![](images/1fe1ac59974dba94381deb02b2fee33a6e96d95e7be525896ff56dbdb92da549.jpg)  
(a) Qwen3.5-4B, ShortGPT

![](images/1b6d9673ccca5a96919ecc0e2cae676cd6bbe9dfd3f3f795100da0920545175a.jpg)  
(b) Qwen3.5-4B, Angular

![](images/a4cf80e59ff5eb8f8e18c6f2c2506d73293bfa0e4d054908f6b12ed198d1a555.jpg)  
(c) Qwen3.5-9B, ShortGPT

![](images/6161469d0dfb72fa51a7d12aaa2f1b58ce2ffd53956aa0e21f2faefdded2dde7.jpg)  
(d) Qwen3.5-9B, Angular

![](images/cce0dd6c71bf826a290fda67d1ec349de07c2fcba852700ee0ac56e73180acdd.jpg)  
(e) Qwen3.6-35B-A3B, REAP  
Figure 9: HomeBench: action-component accuracy for the remaining LLM-pruning combinations. Acc together with the four action components $o , d , p ,$ v against pruning ratio. Each subfigure is one LLM and pruning method. Qwen3-4B results are in Figure 2.

## F.2 Robustness Across Seeds

We measure behavioral robustness using three reruns with the same weights and sampling settings (i.e., temperature is 0.7) for unpruned Qwen3.6- 35B-A3B. For a given instance and metric, an outcome is always correct if all three runs succeed, always wrong if all three fail, and flipping if correctness changes across seeds. The flipping share therefore measures cross-seed instability, whereas the always-wrong share measures a seed-invariant failure core.

Figure 22 shows that task difficulty and crossseed instability are not equivalent. HomeBench Medium has the largest borderline band (24.1% flipping), followed by Partially Executable (18.4%), while Infeasible is comparatively stable (7.1%). In contrast, HAR Medium flips on only 4.5% of instances but is always wrong on 79.7%: it is chronically difficult rather than seed-sensitive. On SHTC, Medium and Complex flip more often than Simple (12.8% and 11.6% versus 6.3%).

![](images/fdf24522a230b4ecc1e764486bf63d751e8f7df2d7c01a87505fe99f89266705.jpg)  
(a) Qwen3-4B, ShortGPT

![](images/ebd6626560d5e9de2e0e2c3ec67150a133e4aa9fbb86ed66ecd93101700d9399.jpg)  
(b) Qwen3-4B, Angular

![](images/6d76431ffced94b962d62c2e23b62152f9ea34761df1cec093cf85e0610779a5.jpg)  
(c) Qwen3-4B, FLAP

![](images/6b19c5a1ce34bbe209f3b8cc1c968f7b3fa98a7f6cddce2333c24e8440ba3cad.jpg)  
(d) Qwen3-4B, 2SSP

![](images/15a766e314ae5307a50f38255d1705c9009c1e9e3d704b30806cdbbe07b1a532.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/2b1c1093c893097ecc82976d074d58c088c07b1c2042e15701807ec58ac44375.jpg)  
(f) Qwen3.5-4B, Angular

![](images/28fee1ecdd9d549fdb10ff649144cf82ce65503fd7bffaac67fff915fb82dd3d.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/54e23c59ad348152f7f85d262ac1775841c840e27715e8bece620412a09021ab.jpg)  
(h) Qwen3.5-9B, Angular

![](images/c72e5203c75d3bd87b4dd39d869a2cf9f5b3074c82662b0d9cb1dc28aea4dfaf.jpg)  
(i) Qwen3.6-35B-A3B, REAP  
Figure 10: HAR: action-component accuracy. Acc together with the four action components $o , d , p ,$ v against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/b9bbcdf5cddf713b07477b632122efe34c86ab8dc60fdf6b9361171c636a2dd4.jpg)  
(a) Qwen3-4B, ShortGPT

![](images/4da8c663ce0d0167982ba003ab3c431f325ec01c8b12bfb5f12c4431d98bb140.jpg)  
(b) Qwen3-4B, Angular

![](images/a6625c8644a45b709b9905ea50b0d989b1581af51e19f0b4318a1937067ce1db.jpg)  
(c) Qwen3-4B, FLAP

![](images/52edc1b6b2c5016ab64b94f806c069777cec8a82099a8558da46c8bf92722653.jpg)  
(d) Qwen3-4B, 2SSP

![](images/22c9f3d584dbd876df9841de5ac14b8a46be7c227ee412fa6edeb67661414457.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/36519039c27d4dec3a16e2a7f5568ce792a9e9f647bcc700be32b4e5929bd19d.jpg)  
(f) Qwen3.5-4B, Angular

![](images/54b8832073a6bc070e0cfbbd5dbff6023eeed048ca62b38b6edfb1b47177c481.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/5ab95936bba30152889434b31cebe819519075a59aeae76cc92121ed9512a28b.jpg)  
(h) Qwen3.5-9B, Angular

![](images/7b8234605228f369a118806b6179f463c6ff6384214797e5f379fbe41fddb45e.jpg)  
(i) Qwen3.6-35B-A3B, REAP  
Figure 11: SHTC: action-component accuracy. ∆Acc relative to $M _ { 0 }$ together with the four action components $o , d , p ,$ v against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/7ad07498aa68eefaac8929da9be6ef3cadaf0abeca74a0a214939ba94052ceb3.jpg)  
(a) Qwen3.5-4B, ShortGPT

![](images/f6356523f9cf825a521b1ebced3a6ca5e20b18bbccf8887db6c6a807799cee6f.jpg)  
(b) Qwen3.5-4B, Angular

![](images/2b2c51ca197bcf276521dd554bd8a18d8ed6dcf1f2c8337f330b8d038fda04a6.jpg)  
(c) Qwen3.5-9B, ShortGPT

![](images/b601998d6bec669df93797c0b4c3cfcfce91255ffd50e36d31adcca03cff1d96.jpg)  
(d) Qwen3.5-9B, Angular

![](images/e218eff706931730a3b635cb316b1963dcff48412a8b47beda01219f5ab368f5.jpg)  
(e) Qwen3.6-35B-A3B, REAP  
Figure 12: HomeBench: accuracy of task-complexity categories for the remaining LLM-pruning combinations. $\operatorname { A c c } _ { c }$ for the complexity categories present in the source against pruning ratio. Each subfigure is one LLM and pruning method. Qwen3-4B results are in Figure 3.

![](images/7d919ae8201bae777e4cadaa100775638971d4576c759bfceacefce3fb564f34.jpg)  
(a) Qwen3-4B, ShortGPT

![](images/d69558b995a0f22297d486fd9ec23b96a8a2f979b2060705fc133af7ba0f4521.jpg)  
(b) Qwen3-4B, Angular

![](images/be902678b475befc3deae8d6fab46b09e5dfa52bcbfc21a88b271bcb81fde32b.jpg)  
(c) Qwen3-4B, FLAP

![](images/97633dbe65aee2b662a4a449be05acc8dfdc2e52a42421621f105caa1e958b96.jpg)  
(d) Qwen3-4B, 2SSP

![](images/31321d37828de5640c67f06f4acf19254549f0df112a099681992160dbf4a1bb.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/1caeddcc2ec802fd556e54b0efbdb79b9f959f0608c60b7978bd3eb27408b9c2.jpg)  
(f) Qwen3.5-4B, Angular

![](images/49b9b4d305832b4dd73d38447252340bb7df52d278ef62ec9065baede7afd964.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/25ba1411f5dbde972789c1c83a8954c032026d668a9e49d224f06f46efd429be.jpg)  
(h) Qwen3.5-9B, Angular

![](images/8b42f40b025600260b40d2e11c1f716082a7a6de228b4369767f33ddabef6c55.jpg)  
(i) Qwen3.6-35B-A3B, REAP

Figure 13: HAR: accuracy of task-complexity categories. $\operatorname { A c c } _ { c }$ for the complexity categories present in the source against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/5bd91646e14e22374e858b4648d0042954ec7dbceb699f49cddcdb892f9e4fd8.jpg)  
(a) Qwen3-4B, ShortGPT

![](images/1e687601f2cd1a7e20d17d2071a8d242c953b701da1869ae4ae2bb1cd7f19386.jpg)  
(b) Qwen3-4B, Angular

![](images/1e80856642ce4d4d2f57d566b78902a4c2dc3aa4956deaa7cc32390ff816e525.jpg)  
(c) Qwen3-4B, FLAP

![](images/9476be64cd854a004cf315c1e0538a3cfe57f36aaa96ddf11c8081d9f46a36bf.jpg)  
(d) Qwen3-4B, 2SSP

![](images/e513b0c5a02689a05222a6587d2287aa02f4747781c8fc461a94ad8c8afc3375.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/03109ead43d93d0ec43d110ec2e1b18fdd914233d41b89691bb4a4dcb6051b7a.jpg)  
(f) Qwen3.5-4B, Angular

![](images/140f3715276ed59906d820d875641318d2dd9cd608dc773c14a333ad6ec0c2ed.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/ddd63934d2c702c9a171c263daec477d3813a593064eccddef72259baa36faaf.jpg)  
(h) Qwen3.5-9B, Angular

![](images/382fe7bfb1c7181f75c22e50ec4bb2160a81298223e4cd3b776aad17c23c6c35.jpg)  
(i) Qwen3.6-35B-A3B, REAP

Figure 14: SHTC: accuracy of task-complexity categories. ∆Acc relative to $M _ { 0 }$ for the complexity categories present in the source against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/f85887c4ba9b6fcab118b8c7ac31a3688bb5d57b6bebc45ad1ca447ed2ae1788.jpg)  
Pruning Ratio (%)  
(a) Qwen3-4B, ShortGPT

![](images/7de01197ee5991c58e173be6b83d0922a31ac6aa1e4444464b8121aeea3cc556.jpg)  
(b) Qwen3-4B, Angular

![](images/f0e35a3c9518905dfd26dea2b6f280273ec84b2011bf12f1a7831de5b461438e.jpg)  
Pruning Ratio (%)  
(c) Qwen3-4B, FLAP

![](images/cc7674da722cdea3463136c86c6dab33cbb96eeed75c066f8c6e6401fd8a159a.jpg)  
(d) Qwen3-4B, 2SSP

![](images/dd014ab25268fb4704c8b82cadbccf4a7254a98fec6c3ffa54d390bc1ee66117.jpg)

![](images/f071914f6fca9dbf5f65901b5a1159d52ccf150f207b72999bf0c54256a24b8f.jpg)  
(f) Qwen3.5-4B, Angular

![](images/a9757da6ec49772f86858ed32e5ffa0a3a9b18df865ec06429017d80cacadaa6.jpg)  
(g) Qwen3.5-9B, ShortGPT

(e) Qwen3.5-4B, ShortGPT  
![](images/5801ade8fa6b5b77068e994fe2144ca365665f966da88069f954bbd8932f3c62.jpg)  
(h) Qwen3.5-9B, Angular

![](images/4c3a9801748beac157332d7a660377c67bd3f4f981b856052632f1c29bdcd996.jpg)  
(i) Qwen3.6-35B-A3B, REAP

Figure 15: HomeBench: operation-level F1 of the action components. F1 together with the four action components o, d, p, v against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/08210c76cf5344748c1164960c6f83784e79136e9200d60477b39c05159a07cc.jpg)  
Pruning Ratio (%)  
(a) Qwen3-4B, ShortGPT

![](images/13ed28627b13692e194a00ab5a77e15d95671fb0a388f2c357993612d67b9531.jpg)  
(b) Qwen3-4B, Angular

![](images/e8236b138468e49491dbb340b7cbe6ceeecfa398f92e7856271692fd717f18d9.jpg)  
Pruning Ratio (%)  
(c) Qwen3-4B, FLAP

![](images/8e88888db47a2c7f861d9a68f5504b7db51d5a69b1ad23c564cb580aab0176b7.jpg)  
(d) Qwen3-4B, 2SSP

![](images/8d2c0e26f9223b68c8e5624ce84336a43b1a5465ace976757adb12806aaee50c.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/e62df1ee7a72b69a34bcf3667e15982c9ce256b5f730f4834481d0ecac3bf8e1.jpg)  
(f) Qwen3.5-4B, Angular

![](images/b9a958a5c0a3be9722ab2fbfa2a7dc1c7297346109ae353f22a65159e360b75b.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/e4121460c636c4fcf1c3644fa08c96c3db45697baf6428841d70f233c134368c.jpg)  
(h) Qwen3.5-9B, Angular

![](images/0e04ca4c3c686ac0e608d6fc6de26618f1f738700016e1f07ef19af2ddfcfc28.jpg)  
(i) Qwen3.6-35B-A3B, REAP

Figure 16: HAR: operation-level F1 of the action components. F1 together with the four action components o, d, p, v against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/8df41178957cd37189220c12c43c77c75a9b9fc8539f38b2fa4891d18d5d1900.jpg)  
(a) Qwen3-4B, ShortGPT

![](images/a58069976844d8d9f6d8ec0103e36a4dde0782d54d5b7b59fb3578d396403d6d.jpg)  
(b) Qwen3-4B, Angular

![](images/7550c2bd0a8150951d962ef18602e02b8af9659e685cd45a190e6e4c4188baa8.jpg)  
(c) Qwen3-4B, FLAP

![](images/20ac0036dfa183055bde3899fbf0ca909177693257928d5cb9bc7fa35901f452.jpg)  
(d) Qwen3-4B, 2SSP

![](images/95aed0b0e30ccc3038a02065f211e48befd28e4466a4f4fd9cd4be6171dc7780.jpg)

![](images/6d82e62c574b5cc2d26b912a347db9a4081b595ef2404e82a4400f352a3955cc.jpg)  
(f) Qwen3.5-4B, Angular

(e) Qwen3.5-4B, ShortGPT  
![](images/0cabf53c0482cbc8b4ab4bac3627793458f6639a9ef035b4d3ee33af3e5ebd33.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/f266c76e421053e72ad42e9131316b9dcd3ff7b10c9a2daa4c603843f516c932.jpg)  
(h) Qwen3.5-9B, Angular

![](images/e0700eaf9e44ca49859a964397d7ca8cd1f4d60566e6c2a318f623b02b7d84c3.jpg)  
(i) Qwen3.6-35B-A3B, REAP

Figure 17: SHTC: operation-level F1 of the action components. ∆F1 relative to $M _ { 0 }$ together with the four action components $o , d , p ,$ v against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/acd9a4d56403ae3980d3ca377a32be806816cd0ecdd7c4b3807d9da962983816.jpg)  
(a) Qwen3-4B, ShortGPT

![](images/f4e79c5e899167ebe8c53803a4991fb9e131d3f84057a78f4e1b4a8a8ffa5e4d.jpg)  
(b) Qwen3-4B, Angular

![](images/e999976d991b223d281fd1cda0c2c8a36c58055740e52c18b1996a2fa92574a7.jpg)  
(c) Qwen3-4B, FLAP

![](images/e2d0d86c5bc6b2206229bfd6efe039978b1f3fff8336f252de0ed2c3ccdcd4dc.jpg)  
(d) Qwen3-4B, 2SSP

![](images/7836fb2e3ccc4a9ffcf0698d251cae35e6f4d91637e8df81d9850340ec787505.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/ee828a4f14be67ea17f2d11c4c576d99e859b489f4f6dc502a34adb00efbf7fd.jpg)  
(f) Qwen3.5-4B, Angular

![](images/33e9fd4043a43fe63e2640657f6c5d7bcfdb881e3f38b2e883da6f6294b84070.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/ffcc12bcae30ce43a6d3709347ca8ad6655eeef6c460641e9ae3331961180a6d.jpg)  
(h) Qwen3.5-9B, Angular

![](images/411a137ecae9025959c121a56fd22d27a3f42d0b587126b75ab349aaaf9c8276.jpg)  
(i) Qwen3.6-35B-A3B, REAP  
Figure 18: HomeBench: operation-level F1 of the task-complexity categories. $\mathrm { F } 1 _ { c }$ for the complexity categories present in the source against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/f98f9c823f85f5098f65c4e5ec0a8215c0fd8204c8c113dc0b0c6eba26eefed1.jpg)  
(a) Qwen3-4B, ShortGPT

![](images/76703af2ed97ee2f29f7af7cdca7b9315aefd21a007cf0079e025d5490883362.jpg)  
(b) Qwen3-4B, Angular

![](images/afc1d8d5112bc3e18f114e5ea09293d14971ea1ce14642b95c937bcbf4a65de6.jpg)  
(c) Qwen3-4B, FLAP

![](images/2989145dc6351ca406981e39261ad7cc1097c9e0cb71723bc51b1daa8fc4609b.jpg)  
(d) Qwen3-4B, 2SSP

![](images/1c9839d7572be1f8d21050a0452a83be884f5da892e229a7e0723af247c8e091.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/c7280835cad3c5447967563c10ec4ad8c07c65e445326771b671f0efa04f39a4.jpg)  
(f) Qwen3.5-4B, Angular

![](images/15bfd5a760261034cdcb644e6c1bffe26a24ae1fc3f75198c975e89a99f8b3f1.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/9eea2010b7f75fd52e1f4f140d497659e89dde4b0dc60d019f79057360971bb1.jpg)  
(h) Qwen3.5-9B, Angular

![](images/4732dc34687fb0b770e65068a1f728398fd7fac8c51fb8b9fafb4394fcc82fea.jpg)  
(i) Qwen3.6-35B-A3B, REAP

Figure 19: HAR: operation-level F1 of the task-complexity categories. $\mathrm { F } 1 _ { c }$ for the complexity categories present in the source against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/555a2b08db2e5be1b24c4498974b21091a4bad5bf0182cbaab6cab51d7beac80.jpg)  
Pruning Ratio (%)  
(a) Qwen3-4B, ShortGPT

![](images/fac09fa8c1e91a95d8dbf6380e218f66a6afd4b08b76c93393038ce154e31ce3.jpg)  
(b) Qwen3-4B, Angular

![](images/e19d6a7037fe9e014c7acb8794cf256d3ccc01a72faccd569dc893c1ce441578.jpg)  
(c) Qwen3-4B, FLAP

![](images/ccb07eb9afa7e1b6ea7907cee5a6665e8373f26d6c13ab329efb941cb2beabd0.jpg)  
(d) Qwen3-4B, 2SSP

![](images/7ef1a7960a47bc95d7c223ea8e6039bd5e17d8390184fb0b3a511a90e84f5f33.jpg)  
(e) Qwen3.5-4B, ShortGPT

![](images/7605c9c51fa04bc24198db26f7e7cfafc30d2f5a7ef569932b1a075eedd71745.jpg)  
(f) Qwen3.5-4B, Angular

![](images/b9535ea1c16957592a84ddd11214e0e55ada9e05a93012962ef059b63054565e.jpg)  
(g) Qwen3.5-9B, ShortGPT

![](images/7e27992771c83300f66b10697563531c08035315c3b0b58a07648fb2fa95256c.jpg)  
(h) Qwen3.5-9B, Angular

![](images/d0f6178598dc8a521d58239099045ea462f3844ae1485daa452389d8605e9faf.jpg)  
(i) Qwen3.6-35B-A3B, REAP

Figure 20: SHTC: operation-level F1 of the task-complexity categories. $\Delta \mathrm { F } 1 _ { c }$ relative to $M _ { 0 }$ for the complexity categories present in the source against pruning ratio. Each subfigure is one LLM and pruning method.

![](images/24c05dae082d242e72f6a251839f3955d7869005b67fca28851928f642c151c5.jpg)  
(a) HomeBench: Action Components

![](images/7f7eb1d18ae3d332cd96f875b66b2699428ea700b51dfdbdcb3dda024d36ed29.jpg)  
(b) HAR: Action Components

![](images/53e591cdd50b0c4c5cfb4996bc8829b90c8429cdc94eb22958b267cb6830a8a6.jpg)  
(c) SHTC: Action Components

![](images/50b4eeb30d28644f9cde1b6456432418d265df26a80e5e172776c8bcbc5b5c53.jpg)  
(d) HomeBench: Task-Complexity Categories

![](images/488a2f434871ec3b0a976640d02c070ce2a8a03995835275b65676abde3374f1.jpg)  
(e) HAR: Task-Complexity Categories

![](images/f721f9e2778309ab63a29816c839cc4e125bdcc7a70470e1bf6cc11251b0f0cc.jpg)  
(f) SHTC: Task-Complexity Categories

Figure 21: Qwen3.6 MoE robustness across REAP pruning ratios by action component and task-complexity level.   
Labels prefixed by C, F, or W above a bar denote Always Correct, Flips Across Ratios, or Always Wrong.

![](images/ed93fbaf11191e5ade907b35e0430ef08626680261bda6cabd00eeecabfccfd2.jpg)  
(a) HomeBench: Action Components

![](images/1c8092889a552f53b9e441d734a35408102574ff51f6d0bdc75208f5a9fa0f37.jpg)  
(b) HAR: Action Components

![](images/26b87b2551fc295cdc922374f1b76452e95a5ae4f6cbdbd489fc3db5300da3bf.jpg)  
(c) SHTC: Action Components

![](images/e787680d5232ff0b5d712dfe20226912ff3cdd010800f01282ce7ff3cb3b70df.jpg)  
(d) HomeBench: Task-Complexity Categories

![](images/cc4d0d314d2787193650d53e2fa79bdaa1f298a6de4f03cd75607da45d7a51bd.jpg)  
(e) HAR: Task-Complexity Categories

![](images/eb38cd6911e84f1cb825053542048977405c08dc478a3555875f8354eb96ac3c.jpg)  
(f) SHTC: Task-Complexity Categories  
Figure 22: Qwen3.6 MoE robustness across seeds by action component and task-complexity level. Labels prefixed by C, F, or W above a bar denote Always Correct, Flips Across Seeds, or Always Wrong.