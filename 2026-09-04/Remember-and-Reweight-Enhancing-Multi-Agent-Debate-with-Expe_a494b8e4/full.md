# Remember and Reweight: Enhancing Multi-Agent Debate with Experience Memory and Confidence Estimation

Xuanfa Jin<sup>1,2</sup> Zhijian Ma<sup>1,2</sup> Yongcheng Zeng<sup>1,2</sup> Xinyu Cui<sup>1,2</sup>

Haifeng Zhang<sup>1,2∗</sup> Jun Wang<sup>3</sup>\*

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences

<sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>University College London {jinxuanfa2022,mazhijian2024,zengyongcheng2022,cuixinyu2021}@ia.ac.cn haifeng.zhang@ia.ac.cn,jun.wang@cs.ucl.ac.uk

## Abstract

Multi-agent debate (MAD) improves the reasoning capabilities of large language models by having multiple agents iteratively refine their responses through discussion. However, MAD suffers from a critical vulnerability known as shared misconception: when a majority of agents initially converge on an incorrect answer, the debate process tends to amplify rather than correct the error. Existing methods primarily address peer skew but leave the agents’ inherently biased concept priors unaddressed. To mitigate this systematic weakness, we propose R<sup>2</sup>-MAD (Remember and Reweight for Multi-Agent Debate), a framework that equips agents with an experience memory accumulated from past debates. R<sup>2</sup>-MAD intervenes on both failure modes through two complementary mechanisms: A debate-state-aware retrieval policy dynamically calibrates the concept prior by retrieving relevant historical evidence based on the current consensus level. Then these retrieved experiences provide a basis for estimating per-agent reliability, yielding confidence weights to modulate peer influence. Experiments on various benchmarks show that R<sup>2</sup>- MAD achieves consistent improvements over existing single-agent and MAD baselines.

## 1 Introduction

Multi-agent debate (MAD) has emerged as a promising paradigm for improving the reasoning capabilities of large language models (LLMs) (Du et al., 2024; Liang et al., 2024). By having multiple LLM-based agents iteratively discuss and refine their responses, MAD has demonstrated consistent gains over single-agent baselines across a range of tasks, including reasoning (Zhu et al., 2026; Ling et al., 2025), evaluation (Chan et al., 2024), and problem solving (Li et al., 2025). The underlying intuition is appealing: diverse agents can challenge each other’s errors, and iterative refinement drives the group toward better answers. However, this optimistic picture obscures a critical vulnerability. When a majority of agents happen to initially converge on the same incorrect answer, which we refer to as shared misconception (Estornell and Liu, 2024), the debate process does not merely fail to correct the error but tends to amplify it. Agents that originally held the correct answer are progressively persuaded to abandon their stances in favor of the majority’s consensus. This failure mode is not an occasional anomaly but a systematic weakness of the MAD framework (Wynn et al., 2025). To mitigate it, we must first understand why debate fails under this regime and which factors contribute to the amplification of the initial error.

![](images/03b5efe30b02b3ea0a3170c6b5a920d5a39bc965a1053d3fe48196b6a5345611.jpg)

![](images/c9562ae89d62d823ea1b556e75fbecfcadbfdd0828b1d43816db8111d0900cc2.jpg)  
Figure 1: An illustrative example of the shared misconception problem. In standard MAD (top), the two agents holding the incorrect answer persuade the correct agent to switch, resulting in a wrong consensus. In R<sup>2</sup>- MAD (bottom), retrieved experiences and confidence weighting help agents resist the erroneous majority and converge to the correct answer.

Recent theoretical analysis provides useful insight into this question. Estornell and Liu (2024) show that under the latent concept framework, each agent’s generation during debate can be decomposed into two factors: a concept prior, which encodes the agent’s intrinsic belief about the correct answer, and a peer skew, which captures the cumulative influence of other agents’ responses. Under shared misconception, these two factors compound: the prior is already biased toward the erroneous concept, and the peer skew amplifies this bias at a rate proportional to the number of agreeing agents. This analysis suggests a natural design principle: effective mitigation should intervene on both factors. But existing approaches (Li et al., 2024; Liu et al., 2024; Tian et al., 2026) focus on manipulating other agents’ responses to reduce the peer skew, leaving the biased prior unaddressed. This motivates us to explore mechanisms that can simultaneously correct the prior and modulate the skew, and we find that experiences from past debates offer a natural substrate for both.

Based on this insight, we propose $\mathbf { R } ^ { 2 } \mathbf { - M A D }$ (Remember and Reweight for Multi-Agent Debate), a framework that equips each agent with an experience memory accumulated from past debates. The memory serves two complementary purposes. First, agents retrieve relevant experiences to calibrate their prior beliefs with historical evidence. The retrieval is guided by a debate-state-aware policy that adapts to the current level of consensus: when consensus is high, retrieval favors diverse and contrastive experiences that challenge the majority; when agents remain divided, retrieval prioritizes experiences associated with positive outcomes. Second, by examining how each agent’s stance has performed in historically similar situations, the retrieved experiences provide a basis for estimating per-agent reliability, which is then used as confidence weights to modulate peer influence during debate. The two mechanisms target the prior and the skew, respectively, and can be naturally combined within a unified framework.

We evaluate $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ on various benchmarks. Experimental results show that $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ outperforms single-agent methods and achieves consistent improvements over existing MAD baselines. Ablation studies confirm that both the memorybased prior correction and the confidence weighting contribute to the overall gains. Our main contributions are summarized as follows: (1) We propose $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ , a framework that mitigates shared misconceptions in multi-agent debate by leveraging experience memory to intervene on both the concept prior and the peer skew. (2) We provide supplementary theoretical analysis that extends the existing latent concept decomposition framework to accommodate memory and confidence weighting, offering formal justification for the proposed design. (3) We conduct extensive experiments on multiple benchmarks, demonstrating the effectiveness of $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ and validating the contribution of each component.<sup>1</sup>

## 2 Related Work

Multi-Agent Debate. The multi-agent debate (MAD) framework leverages collaborative interactions among multiple LLM-based agents to improve reasoning (Du et al., 2024). Following this paradigm, prior work has explored divergent debate (Liang et al., 2024), LLM-as-judge evaluation (Chan et al., 2024), efficient communication structures (Li et al., 2024), and other applications (Chen et al., 2024b; Jin et al., 2024). To further improve debate quality, several methods have been proposed to refine how agents interact: diversity pruning removes near-duplicate responses to encourage broader exploration (Estornell and Liu, 2024), selective masking filters unreliable messages from previous rounds to prevent error propagation (Tian et al., 2026), and FreeMaD replaces rigid turn-taking with flexible, asynchronous updates (Cui et al., 2025). Other studies have diagnosed remaining failure modes, including uncalibrated confidence expression and insufficient agent diversity (Lin and Hooi, 2025; Zhu et al., 2026), as well as costly message passing that motivates equilibrium-based formulations (Yi et al., 2025). Complementary to these efforts, $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ introduces a cross-debate perspective by using experiences accumulated from past debates to calibrate both agent beliefs and peer influence.

Memory-Based LLM Agents. Memory has become an important component of LLM agents. Previous work has used memory to store observations and reflections for more believable longterm behavior (Park et al., 2023), feedback for iterative self-improvement (Shinn et al., 2023), reusable skills acquired from interaction (Wang et al., 2023), and long-term user or interaction histories (Zhong et al., 2024; Packer et al., 2023).

Beyond storing past interactions, recent memoryaugmented methods have improved how memories are retrieved, compressed, and organized, ranging from memory-inspired retrieval (Qian et al., 2024) and question-reflection memory (Wang et al., 2024a) to reversible compression (Wang et al., 2025) and agent-level dynamic memory organization (Xu et al., 2025). When it comes to MAD, $\mathbf { M A D - M } ^ { 2 }$ (Tian et al., 2026) tries to mask unreliable memories from previous debate rounds to prevent error propagation, while MeMAD (Ling et al., 2025) stores structured debate transcripts and retrieves relevant past experiences to guide future reasoning. Compared to them, $\mathbf { R } ^ { 2 } – \mathbf { M A D }$ further leverages retrieved experiences to estimate per-agent reliability for confidence weighting, and dynamically adjusts its retrieval strategy based on the evolving debate state.

## 3 Preliminary

Multi-Agent Debate Framework. Given a task x with a target answer y, the Multi-Agent Debate (MAD) framework (Du et al., 2024) aims to leverage a collective of n LLM-based agents to iteratively resolve the task over $T$ rounds. Let $\phi _ { i }$ denote the parameters of agent i (e.g., model weights, training data, or prior contexts). At the initial round $t = 0$ , each agent i independently generates a response $z _ { i } ^ { ( 0 ) }$ based on the input $x .$ . For subsequent rounds $0 < t \leq T$ , agent i updates its stance by observing the joint responses of all agents from the previous round $Z ^ { ( t - 1 ) } = ( z _ { 1 } ^ { ( t - 1 ) } , \cdot \cdot \cdot , z _ { n } ^ { ( t - 1 ) } )$ Formally, the generation probability of agent i at round t is defined as:

$$
\mathbb { P } \left( z _ { i } ^ { ( t ) } \Big | x , Z ^ { ( { t - 1 } ) } , \phi _ { i } \right) ,\tag{1}
$$

where both the contextual input $( x , Z ^ { ( t - 1 ) } )$ and the internal parameter $\phi _ { i }$ govern the generation process. This iterative debate continues until a maximum horizon $T$ is reached or a consensus is established. Finally, to extract answers from agents’ responses, we define $a ( \cdot )$ as an answer extraction function.

Latent Concept Decomposition. To model the underlying reasoning process in multi-agent debate, the generation dynamics can be analyzed in terms of a latent concept space Θ (Xie et al., 2021). Under this paradigm, each task-answer pair $( x , y )$ is assumed to be generated from a true latent concept $\theta ^ { \star } \in \Theta$ . By introducing a conditional independence assumption, where an agent’s response $z _ { i } ^ { ( t + 1 ) }$ depends solely on the latent concept θ and its parameters $\phi _ { i }$ once θ is given, Estornell and Liu (2024) derives a skew decomposition for the generation probability:

$$
\mathbb { P } \left( z _ { i } ^ { ( t + 1 ) } \Big | x , Z ^ { ( t ) } , \phi _ { i } \right) \propto \sum _ { \theta \in \Theta } \Big [ \mathbb { P } \left( z _ { i } ^ { ( t + 1 ) } \Big | \theta , \phi _ { i } \right)
$$

$$
\mathbb { P } \left( x | \theta , \phi _ { i } \right) \mathbb { P } \left( \theta | \phi _ { i } \right) \prod _ { j = 1 } ^ { n } \mathbb { P } \left( z _ { j } ^ { ( t ) } \Big | \theta , \phi _ { i } \right) \Big ] .\tag{2}
$$

Here, the formulation explicitly factorizes the debate process into two distinct components: the agent’s intrinsic generation capability independent of its peers, and the coordination bias or skew introduced by peer interactions.

## 4 R<sup>2</sup>-MAD Framework

## 4.1 Overview

The latent concept decomposition reveals two distinct factors governing an agent’s generation during debate: the concept prior $\mathbb { P } ( \theta | \phi _ { i } )$ , which encodes the agent’s intrinsic belief, and the peer skew $\Pi _ { j } \mathbb { P } ( z _ { j } ^ { ( t ) } | \theta , \phi _ { i } )$ , which captures the cumulative influence of other agents. When a shared misconception arises, both factors work against correctness. The prior is already biased toward an erroneous concept $\theta ^ { \prime }$ , and the skew amplifies this bias as more agents echo the same mistake.

To address both failure modes, we propose $\mathbf { R } ^ { 2 } .$ MAD (Remember and Reweight for Multi-Agent Debate), which augments each agent with an experience memory populated from past debates. Under a natural conditional independence assumption (detailed in Appendix $\mathrm { C } )$ , the generation probability under our framework is

$$
\begin{array} { r l } & { \mathbb { P } _ { E } \left( z _ { i } ^ { ( t + 1 ) } \Big \vert x , Z ^ { ( t ) } , E _ { i } ^ { ( t ) } , \phi _ { i } \right) } \\ & { \propto \displaystyle \sum _ { \theta \in \Theta } \left[ \mathbb { P } \left( z _ { i } ^ { ( t + 1 ) } \Big \vert \theta , \phi _ { i } \right) \mathbb { P } \left( x \middle \vert \theta , \phi _ { i } \right) \right. } \\ & { \quad \left. \mathbb { P } \left( \theta \Big \vert E _ { i } ^ { ( t ) } , \phi _ { i } \right) \prod _ { j = 1 } ^ { n } \mathbb { P } \left( z _ { j } ^ { ( t ) } \Big \vert \theta , \phi _ { i } \right) ^ { w _ { j , i } ^ { ( t ) } } \right] . } \end{array}\tag{3}
$$

Compared with the vanilla decomposition (Eq. (2)), our proposed framework introduces two modifications. First, the concept prior $\mathbb { P } ( \theta | \phi _ { i } )$ is replaced by a memory-corrected prior $\mathbb { P } ( \boldsymbol { \theta } | E _ { i } ^ { ( t ) } , \phi _ { i } )$ where $E _ { i } ^ { ( t ) } \subseteq M _ { i }$ denotes experiences retrieved from agent $i \ ' _ { \mathrm { { s } } }$ memory bank $M _ { i }$ via a debate-stateaware retrieval policy (Section 4.2). Second, each peer’s contribution to the skew term is modulated by a confidence weight $w _ { j , i } ^ { ( t ) }$ , which is also estimated from agent i’s retrieved experiences (Section 4.3). The two modifications act on distinct components of the decomposition, and the overall framework is illustrated in Figure 2.

![](images/b4df01564174228d8974e72237ae58de3c2cbd8c8e22f16e9424ee4092e3d048.jpg)  
Figure 2: Overview of the R<sup>2</sup>-MAD framework. At each debate round, agent i retrieves relevant experiences from its memory bank via a debate-state-aware policy, uses them to calibrate its prior concepts and estimate confidence weights for each peer, and then generates an updated response.

## 4.2 Debate-State-Aware Memory Retrieval

Retrieving memories based solely on task similarity, as commonly adopted in memory-based LLM agents (Ling et al., 2025; Tian et al., 2026), cannot capture the dynamics of an ongoing debate or adapt to agents’ evolving needs across rounds. For instance, when a majority of agents have agreed on the same answer, the consensus may reflect a shared misconception instead of correctness, and the most valuable memories are those in which a similar majority turned out to be wrong. Conversely, when agents remain divided, the priority shifts to retrieving memories that are both relevant and historically associated with the positive outcomes, helping agents identify the potential right direction. Hence, we first define agent i’s debate state to characterize its current status, and then design a debate-state-aware retrieval policy that dynamically adjusts its strategy accordingly.

Debate State. Before generating a response at round $t \left( t > 0 \right)$ , each agent i observes the current debate state, defined as the tuple

$$
s _ { i } ^ { ( t ) } : = \langle x , z _ { i } ^ { ( t - 1 ) } , h ^ { ( t - 1 ) } , \mathrm { C o n s } ( Z ^ { ( t - 1 ) } ) \rangle ,\tag{4}
$$

where x is the current task, $z _ { i } ^ { ( t - 1 ) }$ is agent i’s previous response, $h ^ { ( t - 1 ) }$ is a natural language summary of the previous round’s debate, and $\mathrm { C o n s } ( Z ^ { ( t - 1 ) } )$ 1 is the consensus ratio, defined as the fraction of agents agreeing on the most popular answer:

$$
\operatorname { C o n s } ( Z ) = \operatorname* { m a x } _ { y } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \left[ a ( z _ { i } ) = y \right] .\tag{5}
$$

Experience Memory. After a debate concludes, each agent i will extract key information from each round t to construct memory cases $e _ { i } ^ { ( t ) }$ and store them in its memory bank $M _ { i }$ . Specifically, the structure of round t’s memory case $\bar { e } _ { i } ^ { ( t ) }$ is formally defined as:

$$
e _ { i } ^ { ( t ) } : = \langle s _ { i } ^ { ( t ) } , Z ^ { ( t ) } , y , \zeta _ { i } ^ { ( t ) } , r _ { i } \rangle .\tag{6}
$$

Here, $Z ^ { ( t ) }$ is all agents’ responses at round t, y is the answer to task x, $\zeta _ { i } ^ { ( t ) }$ indicates the correctness of agent $i \ ' _ { \mathrm { { s } } }$ current response, and $r _ { i }$ is the outcome reward for agent i (1 for correct, 0 for wrong). We note that $r _ { i }$ only requires a signal indicating whether a concluded debate reached a good answer, which any method that learns from debate outcomes necessarily needs. This signal is not tied to gold annotations: it can equally be provided by a verifier model or by execution feedback on verifiable tasks. In our experiments, it simply reuses the labels already in the benchmarks, requiring no annotation beyond the datasets themselves. Over time, the memory bank $M _ { i }$ accumulates a growing collection of debate experiences that can be drawn upon in future debates.

Retrieval Policy. Given the current debate state $s _ { i } ^ { ( t ) }$ and memory bank $M _ { i } .$ , the state-aware retrieval policy proceeds in two stages. Assuming to retrieve $K$ cases from the memory, we first sample a candidate set $\mathcal { C } _ { i } ^ { ( t ) }$ by retrieving top-3K cases based on the cosine similarity between $s _ { i } ^ { ( t ) }$ and the debate state in memory cases. We then select $K$ cases from $\mathcal { C } _ { i } ^ { ( t ) }$ via Maximal Marginal Relevance (MMR), starting from $E _ { i } ^ { ( t ) } = \emptyset$ and greedily selecting at each step:

$$
e ^ { \star } = \arg \operatorname* { m a x } _ { e \in \mathcal { C } _ { i } ^ { ( t ) } \backslash E _ { i } ^ { ( t ) } } \left[ \lambda ^ { t } \cdot \sin ( e , s _ { i } ^ { ( t ) } ) \cdot r _ { i } ( e ) \right.
$$

$$
\left. \begin{array} { r l } { \displaystyle } & { \cdot ( 1 - \lambda ^ { t } ) \operatorname* { m a x } _ { e ^ { \prime } \in E _ { i } ^ { ( t ) } } \mathrm { s i m } ( e , e ^ { \prime } ) \right] , } \end{array}\tag{7}
$$

until the size of the retrieved memory set $| E _ { i } ^ { ( t ) } | =$ $K$ . Here, $r _ { i } ( e )$ is the outcome reward for memory case $e ,$ and the trade-off coefficient $\lambda ^ { t }$ is a function of the consensus ratio:

$$
\begin{array} { r } { \lambda ^ { t } = 1 - \gamma \operatorname { C o n s } ( Z ^ { ( t - 1 ) } ) , \gamma \in [ 0 , 1 ] . } \end{array}\tag{8}
$$

The first term in Eq. (7) combines relevance with outcome quality: the sim $. ( e , s _ { i } ^ { ( t ) } ) \cdot r _ { i } ( e )$ assigns high scores to experiences that are both semantically related and positive. The second term penalizes redundancy by discouraging experiences similar to those already selected. The consensusdependent $\lambda ^ { t }$ dynamically balances the two objectives: 1) when consensus is low, $\lambda ^ { t }$ is large and retrieval favors relevant, positive experiences to help agents reach consensus; 2) when consensus is high, $\lambda ^ { t }$ decreases, allowing diverse and contrastive experiences to enter the retrieved set.

From a theoretical perspective, the retrieved experiences act as a likelihood-ratio correction to the agent’s concept prior, updating $\mathbb { P } ( \theta | \phi _ { i } )$ to $\mathbb { P } ( \boldsymbol { \theta } | E _ { i } ^ { ( t ) } , \phi _ { i } )$ . We formalize this as follows (refer to Appendix C.1 for detailed proof):

Proposition 4.1 (Memory as prior correction). The memory-corrected prior satisfies

$$
\mathbb { P } ( \boldsymbol { \theta } | E _ { i } ^ { ( t ) } , \boldsymbol { \phi } _ { i } ) = P ( \boldsymbol { \theta } | \boldsymbol { \phi } _ { i } ) \cdot \frac { \mathbb { P } ( E _ { i } ^ { ( t ) } | \boldsymbol { \theta } , \boldsymbol { \phi } _ { i } ) } { \mathbb { P } ( E _ { i } ^ { ( t ) } | \boldsymbol { \phi } _ { i } ) } .\tag{9}
$$

That is, the retrieved experiences $E _ { i } ^ { ( t ) }$ act as a likelihood-ratio correction to the agent’s bare prior. In particular, if $E _ { i } ^ { ( t ) }$ provide more evidence for the true concept $\theta ^ { \star }$ than for an erroneous concept $\theta ^ { \prime } { } _ { ; }$ , i.e., $\mathbb { P } ( E _ { i } ^ { ( t ) } | \theta ^ { \star } , \phi _ { i } ) \ >$ $\mathbb { P } ( E _ { i } ^ { ( t ) } | \theta ^ { \prime } , \phi _ { i } )$ , then

$$
\frac { \mathbb { P } ( \theta ^ { \star } | E _ { i } ^ { ( t ) } , \phi _ { i } ) } { \mathbb { P } ( \theta ^ { \prime } | E _ { i } ^ { ( t ) } , \phi _ { i } ) } > \frac { \mathbb { P } ( \theta ^ { \star } | \phi _ { i } ) } { \mathbb { P } ( \theta ^ { \prime } | \phi _ { i } ) } ,\tag{10}
$$

directly counteracting the bias toward $\theta ^ { \prime } .$

## 4.3 Memory-Derived Confidence Weighting

In a standard multi-agent debate, all agents’ responses are presented equally to their peers, regardless of their reliability. However, an agent whose current stance has historically led to negative outcomes in similar circumstances should carry less influence, while an agent whose stance has been consistently positive deserves greater weight. Rather than requiring an external oracle to judge agent quality, we leverage the same retrieved experiences from Eq. (7) to estimate each agent’s reliability and weight their influence accordingly.

Confidence Estimation. Given agent $i \ ' _ { \mathrm { { s } } }$ retrieved experiences $E _ { i } ^ { ( t ) } = \{ e _ { i , 1 } , . . . , e _ { i , K } \}$ , since these experiences share similar debate states with agent i (according to the retrieval policy), we estimate the reliability of agent $j ^ { \circ } \mathbf { s }$ current stance by comparing it against its historical stances recorded in each experience. Specifically, for each experience $e _ { k } .$ , we compute a soft matching weight based on the cosine similarity between agent $j ^ { \dagger } \mathbf { s }$ current response and its historical responses in the retrieved experiences, then aggregate these weights with its correctness to obtain a confidence score:

$$
c _ { j , i } ^ { ( t ) } = \frac { \sum _ { k = 1 } ^ { K } \sin ( z _ { j } ^ { ( t ) } , z _ { j } ( e _ { i , k } ) ) \cdot \zeta _ { j } ( e _ { i , k } ) } { \sum _ { k = 1 } ^ { K } \sin ( z _ { j } ^ { ( t ) } , z _ { j } ( e _ { i , k } ) ) } ,\tag{11}
$$

where $z _ { j } ( e _ { i , k } )$ and $\zeta _ { j } ( e _ { i , k } )$ denote agent $j ^ { \flat } \mathbf { s }$ response and correctness in agent $i \ ' s$ experience $e _ { i , k }$ . Intuitively, $c _ { j , i } ^ { ( t ) }$ reflects agent $i \ ' _ { \mathrm { { s } } }$ estimate of agent $j ^ { \circ } \mathrm { s }$ reliability: by examining how agent j performed in historically similar debate states, agent i assigns higher confidence when agent $j ^ { \prime } { \bf s }$ past responses in similar situations were more often correct, and lower confidence otherwise.

Confidence Weighting. The confidence score determines how much influence agent $j$ should exert in agent i’s subsequent generation. Formally, $c _ { j , i } ^ { ( t ) }$ is mapped to a confidence weight $w _ { j , i } ^ { ( t ) }$ via a sigmoid function such that $c _ { j , i } > 0 . 5$ yields $w _ { j , i } > 1$ (amplified influence), and $c _ { j , i } < 0 . 5$ yields $w _ { j , i } < 1$ (suppressed influence). Since the token probabilities of a black-box LLM cannot be edited directly, we realize $w _ { j , i } ^ { ( t ) }$ at the prompt level by annotating each agent’s response based on two thresholds $w _ { h }$ and $w _ { l }$ (set to 0.55 and 0.45 by default). When $c _ { j , i } ^ { ( t ) } > w _ { h }$ , agent $j ^ { : }$ ’s response is marked by agent i as high confidence, signaling that this stance has been historically reliable in similar debate states. When $c _ { j , i } ^ { ( t ) } < w _ { l }$ , the response is marked as low confidence, indicating limited historical support. Otherwise, the historical evidence is considered inconclusive, and agent $j ^ { \circ } \mathbf { s }$ response is presented without explicit confidence annotation.

Estornell and Liu (2024) show that when m agents produce responses aligned with a common erroneous concept $\theta ^ { \prime } .$ , the debate posterior $\mathbb { P } ( z _ { i } ^ { ( t ) } | x , Z ^ { ( t ) } , \phi _ { i } )$ becomes dominated by $\theta ^ { \prime }$ at a rate of $O ( m )$ , a phenomenon termed majority dominance. Our confidence weighting mechanism can mitigate this effect (proof in Appendix C.2):

Proposition 4.2 (Anti-majority dominance). Suppose m agents share a common misconception $\theta ^ { \prime }$ and are assigned confidence weight $\alpha \in ( 0 , 1 )$ , while the remaining agents receive weight 1. The rate at which the debate posterior converges to $\theta ^ { \prime }$ is slowed from O(m) to O(αm).

We further show that the memory lift and confidence lift are additive in log-odds space, providing a joint guarantee that, under the stated conditions, combining both mechanisms is at least as effective as using either alone. The formal statement and proof are provided as Theorem C.3 in Appendix C.3.

## 5 Experiments

## 5.1 Experiment Setups

Benchmarks. To comprehensively evaluate the performance of $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ , we conduct experiments on four benchmarks that cover both reasoning and knowledge-intensive tasks: MATH500 (Lightman et al., 2024) for mathematical reasoning, Economics and Engineering subsets from MMLU-Pro (Wang et al., 2024b) for domain-specific knowledge reasoning, and TruthfulQA (Lin et al., 2022) for factual judgment. This combination allows us to assess whether $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ generalizes across tasks with different characteristics and patterns.

Baselines. We compare our methods against the following baselines: (1) Chain of Thought (CoT) (Wei et al., 2022), where a single agent performs chain-of-thought reasoning; (2) Self-Consistency (Wang et al., 2022), which samples multiple reasoning paths from a single agent and selects the answer by majority voting; (3) MAD (Du et al., 2024), the standard multi-agent debate framework; and (4) $\mathbf { M A D - M } ^ { 2 }$ (Tian et al., 2026), which augments MAD with memory masking to filter unreliable information from previous rounds. To further isolate whether our gains simply follow from access to related past experiences, we additionally compare against (5) ICL-CoT, a single-agent baseline that retrieves from the same memory bank as $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ and uses the retrieved experiences as in-context examples; results are reported in Appendix B.5.

Implementation Details. We mainly conduct experiments with three open-source LLMs: Qwen2.5- 7B-Instruct (Yang et al., 2024), Qwen3-8B (Yang et al., 2025), and Gemma-3-4B-IT (Team et al., 2025). To further test whether the benefit persists on larger models, we also evaluate it on Llama-3.3-70B-Instruct (Grattafiori et al., 2024) and GPT-4o-mini (Hurst et al., 2024). For all debate-based methods, we use 3 agents over 3 rounds of debate. To prevent test data from leaking into agents’ memory, we split each dataset into a training set for memory construction and a test set for evaluation. For MATH500, we use Level-5 (hardest) problems as the test set and the remaining as the training set. For other datasets, we randomly sampled a portion of the data as the test set. To ensure that no method benefits from prompt-level differences, the system, task, and debate prompts listed in Appendix D are shared identically across all methods. Further details are available in Appendix A.3.

<table><tr><td>Methods</td><td>MATH500</td><td>Economics</td><td>Engineering</td><td>TruthfulQA</td><td>Average</td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>CoT</td><td>0.530</td><td>0.649</td><td>0.407</td><td>0.675</td><td>0.565</td></tr><tr><td>Self-Consistency</td><td>0.522</td><td>0.668</td><td>0.477</td><td>0.704</td><td>0.593</td></tr><tr><td>MAD</td><td>0.515</td><td>0.645</td><td>0.444</td><td>0.681</td><td>0.571</td></tr><tr><td>MAD-M²</td><td>0.403</td><td>0.635</td><td>0.374</td><td>0.590</td><td>0.501</td></tr><tr><td>R²-MAD (ours)</td><td>0.522</td><td>0.701</td><td>0.481</td><td>0.723</td><td>0.607</td></tr><tr><td colspan="6">Qwen3-8B</td></tr><tr><td>CoT</td><td>0.627</td><td>0.787</td><td>0.465</td><td>0.807</td><td>0.672</td></tr><tr><td>Self-Consistency</td><td>0.671</td><td>0.791</td><td>0.535</td><td>0.825</td><td>0.706</td></tr><tr><td>MAD</td><td>0.821</td><td>0.787</td><td>0.634</td><td>0.843</td><td>0.771</td></tr><tr><td> $\mathbf { M A D - M } ^ { 2 }$ </td><td>0.769</td><td>0.787</td><td>0.667</td><td>0.705</td><td>0.732</td></tr><tr><td> $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M A D } \ ( \mathrm { o u r s } )$ </td><td>0.843</td><td>0.796</td><td>0.638</td><td>0.843</td><td>0.780</td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td> $G e m m a - 3 { - } 4 B { - } I T$ </td><td></td><td></td><td></td></tr><tr><td>CoT</td><td>0.500</td><td>0.446</td><td>0.198</td><td>0.627 0.632</td><td>0.443</td></tr><tr><td>Self-Consistency</td><td>0.530</td><td>0.488</td><td>0.255</td><td></td><td>0.476</td></tr><tr><td>MAD  $\mathbf { M A D - M } ^ { 2 }$ </td><td>0.500</td><td>0.502</td><td>0.243 0.259</td><td>0.692 0.448</td><td>0.484</td></tr><tr><td> $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M A D } \ ( \mathrm { o u r s } )$ </td><td>0.537 0.530</td><td>0.507 0.521</td><td>0.263</td><td>0.705</td><td>0.448 0.505</td></tr></table>

Table 1: Overall accuracy of different methods across four benchmarks using three large language models. Bold and underlined values indicate the highest and second-highest accuracies in each model’s column.

## 5.2 Results and Analysis

We organize our experimental analysis around four research questions: RQ1 examines the overall performance of $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ against existing baselines; RQ2 validates the individual contribution of each proposed component through ablation; RQ3 isolates the contribution of the debate-state-aware retrieval policy; and RQ4 investigates whether $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ MAD is effective under the shared misconception regime, which is the core motivation of this work.

RQ1: Does $\mathbf { R } ^ { 2 } \mathbf { - M A D }$ improve over existing single-agent and MAD baselines? To answer this question, we conduct experiments across four benchmarks and three models. The results are presented in Table 1. $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ achieves the highest average accuracy on all three models, consistently outperforming both single-agent methods and debate baselines, with the largest improvements on the knowledge-intensive benchmarks such as Economics and TruthfulQA and somewhat limited on MATH500, where the answer hinges on a long derivation rather than on domain knowledge. Which baseline comes closest varies by model: MAD is the runner-up on Qwen3-8B and Gemma-3-4B-IT, whereas on Qwen2.5-7B-Instruct Self-Consistency becomes the strongest baseline once its sampling budget is matched to that of the debate methods. We also note that $\mathbf { M A D - M } ^ { 2 }$ performs inconsistently, falling below standard MAD on several benchmarks (e.g., MATH500 and TruthfulQA on Qwen2.5-7B-Instruct), suggesting that its memory masking strategy can sometimes discard useful information. In contrast, $\mathbf { R } ^ { 2 } \mathbf { - M A D } ^ { \prime } \mathbf { s }$ selective retrieval and confidence weighting provide more robust improvements. Beyond these three open-source models, $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ also attains the best average accuracy against CoT and MAD on Llama-3.3-70B-Instruct and GPT-4o-mini, indicating that the benefit is not confined to the smallmodel regime. The full results are in Appendix B.6.

RQ2: Do both components contribute to the improvement? To evaluate the individual contributions of the two core components, we conduct an ablation study by removing each module separately: (1) $\mathbf { R } ^ { 2 } \mathbf { - M A D }$ w/o Confidence, which uses memory retrieval but do not assign confidence weights to all agents’ responses; and (2) $\mathbf { R } ^ { 2 } \mathbf { - M A D }$ w/o Memory, which disables adding retrieved experiences into the task prompt and relies solely on confidence weighting. Results are shown in Table 2. On Qwen2.5-7B-Instruct, the full $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ achieves an average accuracy of 0.607, while removing confidence weighting (w/o Confidence) drops performance to 0.585, and removing memory retrieval (w/o Memory) further drops it to 0.576. Both components contribute positively, and the full model consistently outperforms either ablated variant, confirming their complementary nature. Looking across models, confidence weighting shows a more consistent effect: on Gemma-3-4B-IT, removing confidence causes a notable drop in Economics and TruthfulQA, while removing memory leads to a larger drop in TruthfulQA. On Qwen3-8B, where the base MAD performance is already strong, the individual contributions are smaller, but the full model still achieves better performance. These results suggest that both mechanisms provide distinct benefits and that their relative importance varies with the model and task characteristics.

<table><tr><td>Methods</td><td>MATH500</td><td>Economics</td><td>Engineering</td><td>TruthfulQA</td><td>Average</td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>R²-MAD</td><td>0.522</td><td>0.701</td><td>0.481</td><td>0.723</td><td>0.607</td></tr><tr><td>- w/o Confidence</td><td>0.493</td><td>0.664</td><td>0.465</td><td>0.717</td><td>0.585</td></tr><tr><td>- w/o Memory</td><td>0.493</td><td>0.664</td><td>0.453</td><td>0.693</td><td>0.576</td></tr><tr><td colspan="6">Qwen3-8B</td></tr><tr><td>R²-MAD</td><td>0.843</td><td>0.796</td><td>0.638</td><td>0.843</td><td>0.780</td></tr><tr><td>- w/o Confidence</td><td>0.821</td><td>0.782</td><td>0.638</td><td>0.831</td><td>0.768</td></tr><tr><td>- w/o Memory</td><td>0.836</td><td>0.796</td><td>0.654</td><td>0.843</td><td>0.782</td></tr><tr><td colspan="6">Gemma-3-4B-IT</td></tr><tr><td>R²-MAD</td><td>0.530</td><td>0.521</td><td>0.263</td><td>0.705</td><td>0.505</td></tr><tr><td>- w/o Confidence</td><td>0.522</td><td>0.474</td><td>0.263</td><td>0.699</td><td>0.479</td></tr><tr><td>- w/o Memory</td><td>0.537</td><td>0.502</td><td>0.239</td><td>0.657</td><td>0.484</td></tr></table>

Table 2: Ablation study on $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ . "w/o Confidence" removes confidence weights and annotations from peer responses; "w/o Memory" removes retrieved experiences from the task prompt.

<table><tr><td rowspan=1 colspan=1>Policy</td><td rowspan=1 colspan=1>Qwen2.5-7BEcon. TQA</td><td rowspan=1 colspan=1> $G e m m a \ – 3 – 4 B$ Econ.TQA</td><td rowspan=1 colspan=1> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td rowspan=3 colspan=1>RandomSimilarity-basedPositive-only</td><td rowspan=1 colspan=1>0.6680.705</td><td rowspan=1 colspan=1>0.4960.671</td><td rowspan=2 colspan=1>0.6350.647</td></tr><tr><td rowspan=1 colspan=1>0.664 0.717</td><td rowspan=1 colspan=1>0.5210.685</td></tr><tr><td rowspan=1 colspan=1>0.6860.723</td><td rowspan=1 colspan=1>0.5070.705</td><td rowspan=1 colspan=1>0.655</td></tr><tr><td rowspan=2 colspan=1>Diversity-onlyFixed-λ (0.25)</td><td rowspan=1 colspan=1>0.6820.717</td><td rowspan=1 colspan=1>0.5120.685</td><td rowspan=1 colspan=1>0.649</td></tr><tr><td rowspan=1 colspan=1>0.6760.723</td><td rowspan=1 colspan=1>0.5230.695</td><td rowspan=1 colspan=1>0.654</td></tr><tr><td rowspan=1 colspan=1>Fixed-λ (0.50)</td><td rowspan=1 colspan=1>0.6860.711</td><td rowspan=1 colspan=1>0.5040.697</td><td rowspan=1 colspan=1>0.650</td></tr><tr><td rowspan=1 colspan=1>Fixed-λ (0.75)</td><td rowspan=1 colspan=1>0.684 0.723</td><td rowspan=1 colspan=1>0.528 0.691</td><td rowspan=1 colspan=1>0.657</td></tr><tr><td rowspan=1 colspan=1>Debate-State-Aware</td><td rowspan=1 colspan=1>0.701 0.723</td><td rowspan=1 colspan=1>0.521 0.705</td><td rowspan=1 colspan=1>0.663</td></tr></table>

Table 3: Ablation over retrieval policies. Bold indicates the best result in each column.

RQ3: Does the debate-state-aware policy drive the gain? RQ2 establishes that memory helps, but not that our proposed way of retrieving it is responsible. To isolate the retrieval design, we hold every other component fixed and replace only the policy in Eq. (7) with five alternatives: random selection from the memory bank, similarity-based selection on task similarity alone, positive-only selection restricted to cases with positive rewards, diversity-only selection that drops the relevance term, and fixed-λ variants that keep the MMR tradeoff constant instead of conditioning it on the consensus ratio. Table 3 reports the comparison on two models and two benchmarks.

Our policy attains the highest average accuracy (0.663), ahead of the best fixed-λ setting (0.657) and clearly ahead of random (0.635) and similaritybased (0.647) retrieval. That random retrieval is the weakest confirms that the gain does not follow from merely inserting past cases into the prompt, while similarity-based retrieval, the standard choice in memory-augmented agents, also trails every stateaware variant. Our policy is the best or tied-best in three of the four individual settings, demonstrating its effectiveness across different models and tasks. Taken together, these comparisons indicate that the improvement comes from how memory is selected rather than from its mere availability, and that letting the retrieval trade-off respond to the debate state is what makes the memory bank useful.

RQ4: Is $\mathbf { R } ^ { 2 } \mathbf { - M A D }$ effective under shared misconception? The overall results in RQ1 demonstrate the general effectiveness of $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ , but do not reveal whether the gains come specifically from mitigating shared misconceptions, as our framework is designed to address. To directly evaluate this, we isolate the subset of instances where a majority of agents produced incorrect answers at round 0, which represents exactly the shared misconception regime. This is the most challenging setting for any debate-based method, as the peer skew actively works against the correct answer from the beginning. Figure 3 presents the results on this subset. On Qwen2.5-7B-Instruct, $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ substantially outperforms both MAD and $\mathbf { M A D - M } ^ { 2 }$ across all four benchmarks, with the largest margins on Economics (29.41% versus 17.65% for MAD) and Engineering (27.10% versus 17.42%). A similar pattern holds on Gemma-3-4B-IT, where $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ leads on three out of four benchmarks, with the most notable improvement on TruthfulQA. Importantly, the relative improvements on this subset are considerably larger than those observed in the overall results (Table 1), confirming that the gains are concentrated in the regime $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ targets.

![](images/49ebbb0defb55c214f616af670d73c9440ce3ad21451a99ceb39558305d8c38a.jpg)  
(a) Qwen2.5-7B-Instruct

![](images/5591b1a3240894d876e5952f65e467415de6febeda2f94a7ef0d37ea27bfd7ed.jpg)  
(b) Gemma-3-4B-IT  
Figure 3: Final accuracy of different debate methods with different LLMs on the shared misconception subset, where a majority of agents in standard MAD produced incorrect answers at round 0.

However, final accuracy alone does not show how the debate arrives at its answer, which is what Theorem 4.2 concerns: down-weighting an erroneous majority should make correct agents less likely to abandon their stance. To examine this, we further measure two stance-transition rates on the same subset: the fraction of agent-round transitions in which a correct agent switches to a wrong answer $( \mathbf { C } \to \mathbf { W } )$ and the fraction in which a wrong agent recovers the correct one $( \mathbb { W } \to \mathbf { C } )$ , and compare $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ against its w/o Confidence variant to demonstrate the effectiveness of confidence weighting in mitigating the majority dominance. As illustrated in Table 4, confidence weighting successfully reduces harmful flips on three benchmarks, most notably on Engineering (0.410 to 0.255), and increases recoveries on three of the four, with Economics more than doubling (0.075 to 0.172). Corresponding results for the other two models are given in Appendix B.4.

<table><tr><td>Benchmark Method</td><td></td><td>C→W↓ W→C ↑</td><td></td></tr><tr><td rowspan="2">MATH500</td><td> $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ </td><td>0.222</td><td rowspan="2">0.127 0.095</td></tr><tr><td> $\mathbf { \Pi } _ { - } \mathbf { \Pi } _ { \mathbf { W } } / \mathbf { o } \mathbf { \Pi } \mathbf { C o n f }$ </td><td>0.294</td></tr><tr><td rowspan="2">Economics</td><td> $\begin{array} { l } { { \mathrm { R } } ^ { 2 } { \mathrm { - } } { \mathrm { M A D } } } \\ { { \mathrm { - } } \ { \mathrm { w / o } } \ { \mathrm { C o n f } } } \end{array}$ </td><td>0.157</td><td>0.172</td></tr><tr><td></td><td>0.148</td><td>0.075</td></tr><tr><td rowspan="2">Engineering</td><td> $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ </td><td>0.255</td><td>0.164</td></tr><tr><td>- w/o Conf</td><td>0.410</td><td>0.173</td></tr><tr><td rowspan="2">TruthfulQA</td><td> $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ </td><td>0.208</td><td>0.103</td></tr><tr><td>- w/o Conf</td><td>0.269</td><td>0.089</td></tr></table>

Table 4: Stance-transition rates on the shared misconception subset with Qwen2.5-7B-Instruct. C → W denotes correct agents switching to a wrong answer (lower is better) and $\mathbb { W } \to \mathbb { C }$ the reverse (higher is better).

## 6 Conclusion

In this paper, we propose $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D } .$ a framework that addresses the shared misconception problem in multi-agent debate by leveraging experience memory from past debates. Guided by the observation that shared misconceptions stem from two compounding factors in the debate process, a biased concept prior and an amplifying peer skew, $\textstyle \mathbf { R } ^ { 2 } .$ MAD introduces two complementary mechanisms: a debate-state-aware retrieval policy that calibrates agents’ prior beliefs by dynamically selecting relevant historical evidence, and a memory-derived confidence weighting scheme that modulates peer influence based on estimated agent reliability. Experiments across various benchmarks and models demonstrate that $\mathbf { R } ^ { 2 } – \mathbf { M A D }$ consistently improves over both single-agent methods and MAD baselines, with particularly notable gains under the shared misconception regime. Ablation studies further confirm that both components contribute distinct benefits and that their combination yields the best overall performance.

## Limitations

The main limitation of $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ is that it introduces additional computation on top of standard debate. Each round requires summarizing the current debate state, retrieving experiences, and estimating a confidence score for every peer. This overhead is a bounded constant factor rather than one that grows with task difficulty, yet it is a real cost, and R<sup>2</sup>-MAD is correspondingly less suitable than plain debate for latency-critical deployments.

The framework depends on the accumulated memory being informative for the tasks it is applied to. Building this memory requires an outcome signal for each past debate, which restricts our method to settings where such a signal is available. Currently, the memory is constructed offline and remains fixed at test time, which means its usefulness may degrade when the tasks encountered at deployment diverge substantially from those the memory was accumulated on. This dependence also carries a risk: since retrieved experiences shape agents priors, a memory bank built from systematically mistaken trajectories could reinforce an erroneous consensus rather than correct it. So we recommend auditing the memory bank before applying the framework in consequential settings. And extending the framework to update memory continually during deployment, and to weaker forms of outcome supervision, is left to future work.

## Acknowledgments

Haifeng Zhang was supported in part by the National Natural Science Foundation of China under the Original Exploration Program (Grant No. 72450002).

## References

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2024. Chateval: Towards better llm-based evaluators through multi-agent debate. In International conference on learning representations, volume 2024, pages 9079–9093.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024a. M3- embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the association for computational linguistics: ACL 2024, pages 2318–2335.

Justin Chen, Swarnadeep Saha, and Mohit Bansal. 2024b. Reconcile: Round-table conference improves reasoning via consensus among diverse llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7066–7085.

Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, and Cong Zuo. 2025. Free-mad: Consensus-free multiagent debate. arXiv preprint arXiv:2509.11035.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Forty-first international conference on machine learning.

Andrew Estornell and Yang Liu. 2024. Multi-llm debate: Framework, principals, and interventions. Advances in Neural Information Processing Systems, 37:28938–28964.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

OpenAI: Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛adry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, and 399 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Xuanfa Jin, Ziyan Wang, Yali Du, Meng Fang, Haifeng Zhang, and Jun Wang. 2024. Learning to discuss strategically: A case study on one night ultimate werewolf. Advances in Neural Information Processing Systems, 37:77060–77097.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th symposium on operating systems principles, pages 611–626.

Han Li, Yuling Shi, Shaoxin Lin, Xiaodong Gu, Heng Lian, Xin Wang, Yantao Jia, Tao Huang, and Qianxiang Wang. 2025. Swe-debate: Competitive multiagent debate for software issue resolution. arXiv preprint arXiv:2507.23348.

Yunxuan Li, Yibing Du, Jiageng Zhang, Le Hou, Peter Grabowski, Yeqing Li, and Eugene Ie. 2024. Improving multi-agent debate with sparse communication topology. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7281– 7294.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 17889–17904.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th annual meeting of the association for computational linguistics (volume 1: long papers), pages 3214–3252.

Zijie Lin and Bryan Hooi. 2025. Enhancing multi-agent debate system performance via confidence expression. arXiv preprint arXiv:2509.14034.

Shuai Ling, Lizi Liao, Dongmei Jiang, and Weili Guan. 2025. Memad: Structured memory of debates for enhanced multi-agent reasoning. In Second Conference on Language Modeling.

Tongxuan Liu, Xingyu Wang, Weizhe Huang, Wenjiang Xu, Yuting Zeng, Lei Jiang, Hailong Yang, and Jing Li. 2024. Groupdebate: Enhancing the efficiency of multi-agent debate using group discussion. arXiv preprint arXiv:2409.14051.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. 2023. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao, and Zhicheng Dou. 2024. Memorag: Moving towards next-gen rag via memory-inspired knowledge discovery. arXiv preprint arXiv:2409.05591.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Hongduan Tian, Xiao Feng, Ziyuan Zhao, Xiangyu Zhu, Rolan Yan, and Bo Han. 2026. Multi-agent debate with memory masking. arXiv preprint arXiv:2603.20215.

Bo Wang, He-Yan Huang, Yixin Cao, Jiahao Ying, Wei Tang, and Chong Feng. 2024a. Qrmem: Unleash the length limitation through question then reflection memory mechanism. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pages 4837–4851.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Xiaoqiang Wang, Suyuchen Wang, Yun Zhu, and Bang Liu. 2025. R3mem: Bridging memory retention and retrieval via reversible compression. In Findings of the Association for Computational Linguistics: ACL 2025, pages 4541–4557.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024b. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Andrea Wynn, Harsh Satija, and Gillian Hadfield. 2025. Talk isn’t always cheap: Understanding failure modes in multi-agent debate. arXiv preprint arXiv:2509.05396.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Qwen: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,

Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Xie Yi, Zhanke Zhou, Chentao Cao, Qiyu Niu, Tongliang Liu, and Bo Han. 2025. From debate to equilibrium: Belief-driven multi-agent llm reasoning via bayesian nash equilibrium. arXiv preprint arXiv:2506.08292.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. In Proceedings ofthe AAAI conference on artificial intelligence, volume 38(17), pages 19724–19731.

Xiaochen Zhu, Caiqi Zhang, Yizhou Chi, Tom Stafford, Nigel Collier, and Andreas Vlachos. 2026. Demystifying multi-agent debate: The role of confidence and diversity. arXiv preprint arXiv:2601.19921.

## A Additional Experimental Details

## A.1 Benchmark Details

To comprehensively evaluate the performance of R<sup>2</sup>-MAD, we conduct experiments on four benchmarks that cover both reasoning and knowledgeintensive tasks, which are detailed as follows:

• MATH500: This dataset is a subset of the MATH benchmark, consisting of 500 mathematical reasoning problems categorized by topic and difficulty (Lightman et al., 2024). To ensure a rigorously challenging evaluation, we selected 134 problems with the highest difficulty level (Level 5) from the MATH500 dataset as the test set, while adopting the remaining as the training set for memory construction.

• MMLU-Pro Economics: MMLU-Pro is an advanced benchmark assessing multi-disciplinary language understanding and reasoning across 14 domains (Wang et al., 2024b). In this work, we leveraged the Economics subset, which consists of 844 questions with an expanded 10-option multiple-choice format. During the experiment, the dataset was randomly divided into training and test sets in a 3:1 ratio. Consequently, 633 questions were allocated for experience accumulation and 211 for evaluation.

• MMLU-Pro Engineering: In order to measure the effectiveness of our method in more domains, we also employed the Engineering subset of MMLU-Pro (Wang et al., 2024b). This subset contains 969 challenging questions with a 10- option multiple-choice format. Following the same 3:1 split strategy, we randomly partitioned the dataset, resulting in 727 questions for the training set and 242 questions for the test set.

• TruthfulQA: This is a benchmark designed to evaluate whether language models generate truthful answers to questions that are adversarially crafted to elicit common misconceptions and falsehoods (Lin et al., 2022). To facilitate efficient and automated correctness verification, we adopted the multiple-choice evaluation format provided by the dataset, in which the model selects the correct answer from a set of candidate options. However, TruthfulQA’s multiple-choice variant features a variable number of options per question. To maintain a sufficiently challenging evaluation setting, we filtered for questions with 4 to 9 candidate options, yielding a total of 664 questions. Following the same 3:1 random split strategy, 498 questions were allocated to the training set and 166 to the test set.

## A.2 Baseline Details

To validate the effectiveness of our proposed R<sup>2</sup>- MAD, we adopted the following baselines:

• Chain of Thought (CoT): Chain of Thought prompting enhances the reasoning capability of LLMs by instructing the model to decompose a complex problem into intermediate reasoning steps before arriving at the final answer (Wei et al., 2022). In our experiments, each agent generates a single response using CoT prompting, and the answer is directly extracted from this response.

• Self-Consistency: Self-Consistency improves upon CoT by sampling multiple independent reasoning paths for a given question and aggregating final answers by marginalizing out the reasoning paths (Wang et al., 2022). In our settings, the final answer is selected through majority voting over the sampled responses. And to ensure a fair comparison with multi-agent debate methods, which employ 3 agents over 3 debate rounds, we sampled 9 independent reasoning paths for Self-Consistency. Appendix B.2 reports how the number of reasoning paths affects its final accuracy.

• Multi-Agent Debate (MAD): Multi-Agent Debate employs multiple LLM agents to iteratively refine their responses through multi-round discussion (Du et al., 2024). At the initial round, each agent independently generates a response. In subsequent rounds, each agent observes all peers’ responses from the previous round and updates its answer accordingly. The final answer is also determined by majority voting over the last-round responses. Following the default configuration, we set the number of agents to 3 and the number of debate rounds to 3.

<table><tr><td>Methods</td><td>MATH500</td><td>Economics</td><td>s Engineering</td><td>TruthfulQA | Average</td><td></td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td> $\mathbf { M A D - M } ^ { 2 } \left( \mathbf { S } \right)$ </td><td>0.410</td><td>0.597</td><td>0.346</td><td>0.560</td><td>0.478</td></tr><tr><td> $\mathbf { M A D - M ^ { 2 } \left( O \right) }$ </td><td>0.403</td><td>0.635</td><td>0.374</td><td>0.590</td><td>0.501</td></tr><tr><td colspan="6">Qwen3-8B</td></tr><tr><td>MAD-M2 (S)</td><td>0.836</td><td>0.796</td><td>0.700</td><td>0.711</td><td>0.761</td></tr><tr><td> $\mathbf { M A D - M } ^ { 2 }$  (0)</td><td>0.769</td><td>0.787</td><td>0.667</td><td>0.705</td><td>0.732</td></tr><tr><td colspan="6">Gemma-3-4B-IT</td></tr><tr><td> $\mathbf { M A D - M } ^ { 2 } \left( \mathbf { S } \right)$ </td><td>0.448</td><td>0.550</td><td>0.247</td><td>0.428</td><td>0.418</td></tr><tr><td> $\mathbf { M A D - M ^ { 2 } \left( O \right) }$ </td><td>0.537</td><td>0.507</td><td>0.259</td><td>0.448</td><td>0.448</td></tr></table>

Table 5: Comparison of MAD-M<sup>2</sup> under subjective (S) and objective (O) masking strategies across four benchmarks and three large language models.  
or confidence weighting.

• MAD-M<sup>2</sup>: Multi-Agent Debate with Memory Masking addresses the vulnerability of the standard MAD framework to erroneous memories by introducing an evaluation-and-masking phase between debate rounds (Tian et al., 2026). Before each subsequent round, agents critically evaluate the responses from the previous round and generate a binary mask to filter out potentially incorrect memories. $\mathbf { M A D - M } ^ { 2 }$ supports both a subjective masking, where agents explicitly judge each response, and an objective masking strategy based on response perplexity. In our main experiments, we adopt the objective masking variant with the default configuration of 3 agents and 3 debate rounds.

• ICL-CoT: To test whether the benefit of $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ MAD can be attributed simply to having access to related past cases, we construct a retrievalaugmented single-agent baseline. For each test question, it retrieves cases from exactly the same memory bank that $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ uses and places them in the prompt as in-context examples before performing standard CoT reasoning. Since a single agent has no debate state, retrieval here is keyed on task similarity. This baseline therefore receives the same memory content as $\mathtt { R } ^ { 2 } \mathtt { - }$ MAD but none of its debate-state-aware retrieval

## A.3 Implementation Details

Model Configuration. Experiments are mainly conducted using three open-source LLMs: Qwen2.5-7B-Instruct (Yang et al., 2024), Qwen3- 8B (Yang et al., 2025), and Gemma-3-4B-IT (Team et al., 2025). We access all models via local inference with vLLM (Kwon et al., 2023). To encourage diverse responses, the generation temperature is set to 1.0 for all models. Moreover, the top\_p parameter is set to 1.0 across all models. And the maximum output token length for each response is set to 6144 for all models during both the training and evaluation stages. Other LLM-related hyperparameters are set to their default values. For the two larger models, Llama-3.3-70B-Instruct (Grattafiori et al., 2024) and GPT-4o-mini (Hurst et al., 2024), we use API access instead of local inference, keeping the same 3-agent, 3-round configuration, the same agent personas and prompts, and the same decoding settings where the API exposes them.

Memory Construction. The experience memory is constructed offline before evaluation by running the full debate procedure on the training set of each benchmark. Specifically, for each question, we run a standard 3-agent, 3-round debate using the agent personas and prompts described in Appendix D. To enrich the diversity of collected experiences, we force all agents to continue debating until the maximum number of rounds is reached, regardless of whether consensus has been established. This ensures the memory bank captures a wider range of debate dynamics, including scenarios where agents must break out of a premature consensus. Since the initial round $( t = 0 )$ involves each agent answering independently without access to peer responses or debate state information, we exclude round-0 responses from the memory. For each subsequent round $( t > 0 )$ , we extract a memory case following the structure defined in $\operatorname { E q . } ( 6 )$ and store it in the corresponding agent’s memory bank. In our settings, each agent maintains its own memory bank, reflecting debate trajectories and outcomes from its own perspective. The resulting memory bank for each agent contains approximately $2 N _ { \mathrm { t r a i n } }$ cases per benchmark, where $N _ { \mathrm { t r a i n } }$ is the number of questions in the training set of the benchmark.

<table><tr><td>Methods</td><td>MATH500</td><td>Economics</td><td>Engineering</td><td>TruthfulQA</td><td>Average</td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>Self-Consistency (3)</td><td>0.530</td><td>0.644</td><td>0.457</td><td>0.686</td><td>0.579</td></tr><tr><td>Self-Consistency (6)</td><td>0.507</td><td>0.649</td><td>0.469</td><td>0.704</td><td>0.582</td></tr><tr><td>Self-Consistency (9)</td><td>0.522</td><td>0.668</td><td>0.477</td><td>0.704</td><td>0.593</td></tr><tr><td colspan="6">Qwen3-8B</td></tr><tr><td>Self-Consistency (3)</td><td>0.649</td><td>0.768</td><td>0.490</td><td>0.819</td><td>0.682</td></tr><tr><td>Self-Consistency (6)</td><td>0.649</td><td>0.787</td><td>0.490</td><td>0.831</td><td>0.689</td></tr><tr><td>Self-Consistency (9)</td><td>0.671</td><td>0.791</td><td>0.535</td><td>0.825</td><td>0.706</td></tr><tr><td colspan="6"> $G e m m a - 3 { - } 4 B { - } I T$ </td></tr><tr><td>Self-Consistency (3)</td><td>0.485</td><td>0.498</td><td>0.222</td><td>0.657</td><td>0.466</td></tr><tr><td>Self-Consistency (6)</td><td>0.522</td><td>0.479</td><td>0.259</td><td>0.608</td><td>0.467</td></tr><tr><td>Self-Consistency (9)</td><td>0.530</td><td>0.488</td><td>0.255</td><td>0.632</td><td>0.476</td></tr></table>

Table 6: Accuracy of self-consistency with different numbers of paths across four benchmarks and three models. Bold values indicate the best result in each model’s column.

Debate and Retrieval Procedure. At test time, each debate runs for 3 rounds with 3 agents. At each round $t > 0 ,$ , the retrieval policy first encodes the current debate state into an embedding vector using BGE-M3 (Chen et al., 2024a). A candidate set of 3K memory cases is then retrieved by ranking all entries in the agent’s memory bank by cosine similarity to this embedding and retaining the top-$3 K$ This initial filtering serves two purposes: it reduces the computational cost of the subsequent MMR selection, and it establishes a lower bound ensuring that all final candidates maintain a minimum level of relevance to the current debate state. From this candidate set, $K$ cases are selected via MMR with the consensus-based trade-off coefficient $\lambda _ { t } = 1 - \gamma \cdot \mathbf { C o n s } ( Z ^ { ( t - 1 ) } )$ , where $\gamma = 0 . 9$ For confidence weighting, the thresholds are set to $w _ { h } = 0 . 5 5$ and $w _ { l } = 0 . 4 5$ in practice. And when $c _ { j , i } ^ { ( t ) } < w _ { l } \ \mathrm { o r } \ c _ { j , i } ^ { ( t ) } > w _ { h }$ , agent $j ^ { \prime } { \bf s }$ response will be appended with an annotation enclosed by <confidence></confidence>; otherwise, no annotation is added. The consensus ratio is computed by extracting answers from each agent’s response and calculating the fraction of agents agreeing on the most frequent answer. During evaluation, early stopping is applied if all agents reach consensus before round $T .$ . For the final answer, majority voting is performed over the responses of all agents.

## B Additional Experimental Results

## B.1 Comparison of MAD-M<sup>2</sup> Masking Strategies

In the main experiments (Table 1), we adopt the objective (O) masking variant of MAD-M<sup>2</sup> as the default, since it achieves higher average accuracy across most models. Here, we report the full comparison between the two masking strategies for completeness. Table 5 presents the results of MAD-${ \bf M } ^ { 2 }$ under both the subjective (S) masking strategy, where agents explicitly evaluate each peer response and generate binary masks, and the objective (O) masking strategy, which retains only the response with the lowest perplexity. The results reveal that neither strategy consistently dominates across all settings: their relative effectiveness varies with the model and task, consistent with findings in Tian et al. (2026). Importantly, $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ outperforms both variants across all three models in average accuracy, suggesting that leveraging retrieved experiences for prior correction and confidence weighting provides more robust improvement.

<table><tr><td>Methods</td><td>MATH500</td><td>Economics</td><td>Engineering</td><td>TruthfulQA</td><td>Average</td></tr><tr><td colspan="6">Llama-3.3-70B-Instruct</td></tr><tr><td>CoT</td><td>0.485</td><td>0.754</td><td>0.539</td><td>0.799</td><td>0.644</td></tr><tr><td>MAD</td><td>0.542</td><td>0.750</td><td>0.642</td><td>0.874</td><td>0.702</td></tr><tr><td>R²-MAD</td><td>0.560</td><td>0.766</td><td>0.667</td><td>0.874</td><td>0.717</td></tr><tr><td colspan="6">GPT-4o-mini</td></tr><tr><td>CoT</td><td>0.500</td><td>0.703</td><td>0.391</td><td>0.829</td><td>0.606</td></tr><tr><td>MAD</td><td>0.560</td><td>0.728</td><td>0.429</td><td>0.874</td><td>0.648</td></tr><tr><td> $\mathbf { R } ^ { 2 } \mathbf { - M A D }$ </td><td>0.560</td><td>0.735</td><td>0.486</td><td>0.861</td><td>0.661</td></tr></table>

Table 7: Overall accuracy on two larger models: Llama-3.3-70B-Instruct and GPT-4o-mini. Bold values indicate the best result in each model’s column.

## B.2 Number of Reasoning Paths in Self-Consistency

The compute available to Self-Consistency is set by a free parameter, the number of sampled reasoning paths, so how that parameter is chosen determines how strong a baseline it is. To keep the comparison in Table 1 fair, we set it by matching compute: since all debate-based methods in this work use 3 agents over 3 rounds, we sample 9 reasoning paths so that Self-Consistency issues the same number of generation calls per query. Table 6 reports what this choice costs us by evaluating 3 and 6 paths in addition. Average accuracy increases monotonically with the number of paths on all three models, so the setting we adopt is also the strongest of the three for Self-Consistency on every model. The comparison in Table 1 is therefore against the best-performing configuration of this baseline rather than a weakened one, and $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ still attains higher average accuracy on all three models.

## B.3 Additional Shared Misconception Results

![](images/85a5015a4cded62ceba1c187cdc424431b93af99830fd0f310657acf92facc32.jpg)  
Figure 4: Final accuracy of different debate methods with Qwen3-8B on the shared misconception subset, where a majority of agents in standard MAD produced incorrect answers at round 0.

In Section 5.2 (RQ4), we present the shared misconception analysis for Qwen2.5-7B-Instruct and Gemma-3-4B-IT in Figure 3. Here we provide the corresponding results for Qwen3-8B in Figure 4. On this stronger model, $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ remains effective under the shared misconception regime, leading on three benchmarks. It is interesting that in Economics, all three debate methods struggle with accuracies below 10%, suggesting that when a strong model converges on misconceptions at prior, the errors are particularly resistant to correction. Overall, the Qwen3-8B results reinforce the pattern observed across the other two models: $\mathbf { R } ^ { 2 } \mathbf { - M A D } ^ { \prime } \mathbf { s }$ combination of memory-based prior correction and confidence weighting is effective at mitigating shared misconceptions across models of varying capability.

## B.4 Additional Flip and Recovery Analysis

Here we provide the corresponding results for the other two models in Table 8, measured on the same shared misconception subset and under the same definitions. The pattern observed in Table 4 still holds for both models. For each of them, confidence weighting lowers the rate at which correct agents abandon their stance on three of the four benchmarks and raises the recovery rate in most cases. The effect is most pronounced on Economics, where the flip rate falls from 0.333 to 0.077 on Qwen3-8B and from 0.385 to 0.196 on Gemma-3-4B-IT. Counting across all three models, each of the two rates improves in nine of the twelve modelbenchmark pairs, and the largest differences in the table are consistently in favor of the full model. These stance-transition rates isolate the contribution of confidence weighting, and they move in the direction Theorem 4.2 predicts, consistent with the gains $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$ obtains over the debate baselines on the shared misconception subset.

<table><tr><td>Benchmark Method</td><td></td><td>C→W↓ W→C ↑</td><td></td></tr><tr><td></td><td colspan="3">Qwen3-8B</td></tr><tr><td>MATH500</td><td>R²-MAD - w/o Conf R²-MAD</td><td>0.032 0.038</td><td>0.557 0.524</td></tr><tr><td>Economics</td><td>- w/o Conf</td><td>0.077 0.333</td><td>0.049 0.033</td></tr><tr><td>Engineering</td><td>R²-MAD - w/o Conf</td><td>0.180 0.189</td><td>0.290 0.288</td></tr><tr><td>TruthfulQA</td><td> $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$  - w/o Conf</td><td>0.294 0.222</td><td>0.079 0.133</td></tr><tr><td></td><td>Gemma-3-4B-IT</td><td></td><td></td></tr><tr><td>MATH500</td><td> $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M } \mathbf { A } \mathbf { D }$  - w/o Conf</td><td>0.348 0.393</td><td>0.115 0.117</td></tr><tr><td>Economics</td><td>R²-MAD - w/o Conf</td><td>0.196 0.385</td><td>0.098 0.077</td></tr><tr><td>Engineering</td><td>R²-MAD - w/o Conf</td><td>0.461 0.402</td><td>0.127 0.115</td></tr><tr><td>TruthfulQA</td><td>R2-MAD - w/o Conf</td><td>0.182 0.233</td><td>0.142 0.094</td></tr></table>

Table 8: Stance-transition rates on the shared misconception subset for Qwen3-8B and Gemma-3-4B-IT. C → W denotes correct agents switching to a wrong answer and W → C the reverse.

## B.5 Comparison with a Retrieval-Augmented Single-Agent Baseline

<table><tr><td>Model</td><td>Benchmark</td><td>CoT</td><td>ICL-CoT</td><td> $\mathbf { R } ^ { 2 } \mathbf { - M A D }$ </td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>Economics</td><td>0.649</td><td>0.638</td><td>0.701</td></tr><tr><td>TruthfulQA</td><td>0.675</td><td>0.723</td><td>0.723</td></tr><tr><td rowspan="2">Gemma-3-4B</td><td>Economics</td><td>0.446</td><td>0.501</td><td>0.521</td></tr><tr><td>TruthfulQA</td><td>0.627</td><td>0.619</td><td>0.705</td></tr><tr><td>Average</td><td></td><td>|0.599</td><td>0.620</td><td>0.663</td></tr></table>

Table 9: Comparison against ICL-CoT, a single-agent baseline that retrieves from the same memory bank and leverages the retrieved cases as in-context examples.

A natural concern about any memory-augmented method is that its advantage may come from nothing more than access to related question-answer pairs, in which case a single agent given the same cases should perform comparably. We test this directly with ICL-CoT, which retrieves from the identical memory bank and receives the same number of cases in its prompt, but performs single-agent CoT reasoning rather than debate.

The comparison results are listed in Table 9. $\textstyle \mathrm { \mathrm { R } } ^ { 2 } .$ MAD matches or exceeds ICL-CoT in all four settings and is strictly better in three, showing that the same memory content yields a larger benefit when it is retrieved according to the debate state and used to reweight peer influence than when it is inserted as static exemplars. Indeed, ICL-CoT does not even improve reliably over naive CoT, falling below it on Economics with Qwen2.5-7B-

Instruct and on TruthfulQA with Gemma-3-4B-IT, which indicates that having related cases available is not by itself sufficient. The advantage of $\mathtt { R } ^ { 2 } \mathtt { - }$ MAD therefore does not reduce to access to similar question-answer pairs.

## B.6 Results on Larger Models

The three models used in Section 5 all fall in the 4B– 8B range. To test whether the benefit of $\mathbf { R } ^ { 2 } .$ -MAD persists on stronger models, we further evaluate $\mathbf { R } ^ { 2 } .$ MAD against CoT and MAD on Llama-3.3-70B-Instruct (Grattafiori et al., 2024) and GPT-4o-mini (Hurst et al., 2024), covering both open-source and commercial settings. As shown in Table 7, $\mathbf { R } ^ { 2 } \mathbf { - }$ MAD attains the best average accuracy on both models, improving over MAD from 0.702 to 0.717 on Llama-3.3-70B-Instruct and from 0.648 to 0.661 on GPT-4o-mini, which indicates that the benefit is not confined to the small-model regime. We note, however, that the per-benchmark picture is more mixed than in Table 1: the gains concentrate on Economics and Engineering, while TruthfulQA is matched on Llama-3.3-70B-Instruct and slightly lower than MAD on GPT-4o-mini. A plausible reading is that stronger LLMs already resolve much of what memory would otherwise supply on the more knowledge-saturated tasks, leaving less headroom for prior correction.

## B.7 Computational Cost

<table><tr><td>Model</td><td>Method</td><td>Debate Tok. (#)</td><td>Summary Tok. (#)</td><td>Time (s)</td></tr><tr><td>Qwen2.5-7B</td><td>SC MAD R2-MAD</td><td>5.75K 4.44K 5.41K</td><td>2.25K</td><td>0.62 0.49 0.96</td></tr><tr><td>Qwen3-8B</td><td>SC MAD R2-MAD</td><td>14.67K 6.16K 6.38K</td><td>一 1.85K</td><td>5.65 3.75 6.28</td></tr><tr><td>Gemma-3-4B MAD</td><td>SC R2-MAD</td><td>6.09K 6.99K 7.87K</td><td>2.76K</td><td>0.80 0.81 1.18</td></tr></table>

Table 10: Per-query token consumption and wall-clock time statistics on Economics. SC uses 9 reasoning paths, matching the configuration reported in Table 1.

We characterize the per-query cost of $\mathbf { R } ^ { 2 } .$ -MAD along three sources. Debate and summarization invoke the LLMs and are therefore measured in tokens, whereas retrieval and confidence estimation are purely embedding-based, issue no generation call, so are approximately measured by wall-clock time. Table 10 reports both on Economics, with Self-Consistency (SC) and MAD as references.

The debate token consumption of $\mathbf { R } ^ { 2 } { \cdot } \mathbf { M A D } \exp { - }$ ceeds MAD by roughly 0.2K–1.0K tokens per query, which reflects the fixed number of retrieved memory tokens injected per round; because K is constant, this component does not grow with problem scale. Summarization adds a further 1.9K– 2.8K tokens and is the only extra generation call our framework introduces. In wall-clock terms, $\mathtt { R } ^ { 2 } .$ MAD is roughly $1 . 5 \times { } \mathrm { t o } { 2 } \times { } \mathrm { M A D }$ across the three models and modestly slower than Self-Consistency, so all the methods compared remain within the same order of magnitude. Therefore, the overhead is a bounded constant factor rather than one that grows with task difficulty.

## C Proofs

## C.1 Proof for Proposition 4.1

Proof. Part 1: Prior correction identity. Applying Bayes’ rule to the joint distribution of $( \theta , E _ { i } ^ { ( t ) } )$ conditional on $\phi _ { i }$

$$
\mathbb { P } ( \theta \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) = \frac { \mathbb { P } ( E _ { i } ^ { ( t ) } \mid \theta , \phi _ { i } ) \mathbb { P } ( \theta \mid \phi _ { i } ) } { \mathbb { P } ( E _ { i } ^ { ( t ) } \mid \phi _ { i } ) } ,\tag{12}
$$

which is exactly the claimed identity. Here the denominator is the marginal likelihood $\mathbb { P } ( E _ { i } ^ { ( t ) } \mid \phi _ { i } ) =$ $\begin{array} { r } { \sum _ { \theta ^ { \prime } \in \Theta } \mathbb { P } ( E _ { i } ^ { ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) \mathbb { P } ( \theta ^ { \prime } \mid \phi _ { i } ) } \end{array}$ , which is positive and independent of $\theta ,$ so the identity is well-defined. Part 2: Misconception suppression. Taking the ratio of the corrected priors at $\theta ^ { \star }$ and $\theta ^ { \prime }$

$$
\frac { \mathbb P ( \theta ^ { \star } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } { \mathbb P ( \theta ^ { \prime } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } = \frac { \mathbb P ( \theta ^ { \star } \mid \phi _ { i } ) \cdot \frac { \mathbb P ( E _ { i } ^ { ( t ) } \mid \theta ^ { \star } , \phi _ { i } ) } { \mathbb P ( E _ { i } ^ { ( t ) } \mid \phi _ { i } ) } } { \mathbb P ( \theta ^ { \prime } \mid \phi _ { i } ) \cdot \frac { \mathbb P ( E _ { i } ^ { ( t ) } \mid \theta ^ { \star } , \phi _ { i } ) } { \mathbb P ( E _ { i } ^ { ( t ) } \mid \phi _ { i } ) } } = \frac { \mathbb P ( \theta ^ { \star } \mid \phi _ { i } ) } { \mathbb P ( \theta ^ { \prime } \mid \phi _ { i } ) } \cdot \frac { \mathbb P ( E _ { i } ^ { ( t ) } \mid \theta ^ { \star } , \phi _ { i } ) } { \mathbb P ( E _ { i } ^ { ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) } ,\tag{13}
$$

where the $\mathbb { P } ( E _ { i } ^ { ( t ) } \mid \phi _ { i } )$ terms cancel. The first factor is the bare prior ratio. The second factor is the likelihood ratio of the retrieved experiences under $\theta ^ { \star }$ versus $\theta ^ { \prime }$ . By assumption, $\mathbb { P } ( E _ { i } ^ { ( t ) } \mid \theta ^ { \star } , \phi _ { i } ) >$ $\mathbb { P } ( E _ { i } ^ { ( t ) } \mid \theta ^ { \prime } , \phi _ { i } )$ , so this likelihood ratio is strictly greater than 1. Hence

$$
\frac { \mathbb { P } ( \theta ^ { \star } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } { \mathbb { P } ( \theta ^ { \prime } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } > \frac { \mathbb { P } ( \theta ^ { \star } \mid \phi _ { i } ) } { \mathbb { P } ( \theta ^ { \prime } \mid \phi _ { i } ) } ,\tag{14}
$$

confirming that informative memory retrieval shifts the prior ratio in favor of the true concept $\theta ^ { \star }$ , directly counteracting pre-existing bias toward $\theta ^ { \prime }$ □

## C.2 Proof for Proposition 4.2

Proof. Following the setup of Theorem 5.2 in Estornell and Liu (2024), suppose $Z ^ { ( t ) }$ contains m responses sharing a most-likely concept $\theta ^ { \prime }$ , i.e., for $j \le m , \theta ^ { \prime } =$ arg max<sub>θ</sub> $\mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta , \phi _ { i } )$ . Let $z ^ { \prime ( t ) }$ denote a canonical representative of any such majority response. For any two candidate responses $z _ { ( i , 1 ) } ^ { ( t + 1 ) }$ and $z _ { ( i , 2 ) } ^ { ( t + 1 ) }$ , the ratio of their generation probabilities under confidence weighting is

$$
\frac { \mathbb { P } _ { w } \big ( z _ { ( i , 1 ) } ^ { ( t + 1 ) } \mid Z ^ { ( t ) } , x , \phi _ { i } \big ) } { \mathbb { P } _ { w } \big ( z _ { ( i , 2 ) } ^ { ( t + 1 ) } \mid Z ^ { ( t ) } , x , \phi _ { i } \big ) } = \sum _ { \theta \in \Theta } \frac { \mathbb { P } ( z _ { ( i , 1 ) } ^ { ( t + 1 ) } \mid \theta , \phi _ { i } ) \mathbb { P } ( x \mid \theta , \phi _ { i } ) \mathbb { P } ( \theta \mid \phi _ { i } ) } { \mathbb { P } _ { \theta } \big ( z _ { ( i , 2 ) } ^ { ( t + 1 ) } \mid \theta , \phi _ { i } \big ) \mathbb { P } ( x \mid \theta , \phi _ { i } ) \mathbb { P } ( \theta \mid \phi _ { i } ) } \prod _ { j = 1 } ^ { n } \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta , \phi _ { i } ) ^ { w _ { j , i } ^ { ( t ) } }\tag{15}
$$

Step 1: Separating majority and minority contributions. With $w _ { j , i } ^ { ( t ) } = \alpha \mathrm { f o r } j \le m$ and $w _ { j , i } ^ { ( t ) } = 1$ for $j > m$ , we split the product over agents:

$$
\prod _ { j = 1 } ^ { n } \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta , \phi _ { i } ) ^ { w _ { j , i } ^ { ( t ) } } = \underbrace { \prod _ { j \leq m } \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta , \phi _ { i } ) ^ { \alpha } } _ { \mathrm { m a j o r i t y ~ ( w e i g h t e d ) } } \cdot \underbrace { \prod _ { j > m } \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta , \phi _ { i } ) } _ { \mathrm { m i n o r i t y ~ ( u n w e i g h t e d ) } } .\tag{16}
$$

Since all m majority responses share the most-likely concept $\theta ^ { \prime }$ , we approximate $\begin{array} { r } { \prod _ { j \leq m } \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta , \phi _ { i } ) ^ { \alpha } } \end{array}$ ≈ $\mathbb { P } ( z ^ { \prime ( t ) } \mid \theta , \phi _ { i } ) ^ { \alpha m }$ . Substituting into (15), both numerator and denominator take the form

$$
\sum _ { \theta \in \Theta } \mathbb { P } ( z _ { ( i , \cdot ) } ^ { ( t + 1 ) } \mid \theta , \phi _ { i } ) \mathbb { P } ( x \mid \theta , \phi _ { i } ) \mathbb { P } ( \theta \mid \phi _ { i } ) \prod _ { j > m } \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta , \phi _ { i } ) \cdot \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta , \phi _ { i } ) ^ { \alpha m } .\tag{17}
$$

Step 2: Normalizing by the dominant term. Dividing both numerator and denominator by $\mathbb { P } ( z ^ { \prime ( t ) }$ | $\theta ^ { \prime } , \phi _ { i } ) ^ { \alpha m }$ , each summand corresponding to concept θ acquires the factor

$$
\left( \frac { \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta , \phi _ { i } ) } { \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) } \right) ^ { \alpha m } .\tag{18}
$$

By definition of $\theta ^ { \prime }$ as the concept maximizing $\mathbb { P } \big ( z ^ { \prime ( t ) } \mid \theta , \phi _ { i } \big )$ , for every $\theta \neq \theta ^ { \prime }$ we have

$$
\frac { \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta , \phi _ { i } ) } { \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) } \ : < \ : 1 .\tag{19}
$$

Step 3: Taking $m  \infty .$ . For each $\theta \neq \theta ^ { \prime }$ , the factor $\left( \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta , \phi _ { i } ) / \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) \right) ^ { \alpha m }$ converges to 0 as $m \to \infty$ , since the base is strictly less than 1 and the exponent α $n \to \infty$ . Only the $\theta = \theta ^ { \prime }$ summand (whose factor equals 1) survives. Therefore

$$
\operatorname* { l i m } _ { m  \infty } \frac { \mathbb { P } _ { w } ( z _ { ( i , 1 ) } ^ { ( t + 1 ) } \mid Z ^ { ( t ) } , x , \phi _ { i } ) } { \mathbb { P } _ { w } ( z _ { ( i , 2 ) } ^ { ( t + 1 ) } \mid Z ^ { ( t ) } , x , \phi _ { i } ) } = \frac { \mathbb { P } ( z _ { ( i , 1 ) } ^ { ( t + 1 ) } \mid \theta ^ { \prime } , \phi _ { i } ) } { \mathbb { P } ( z _ { ( i , 2 ) } ^ { ( t + 1 ) } \mid \theta ^ { \prime } , \phi _ { i } ) } .\tag{20}
$$

Step 4: Convergence rate analysis. The rate at which the non- ${ \cdot } \theta ^ { \prime }$ terms vanish is governed by the exponent αm. Specifically, for the leading competing concept $\theta ^ { * } \neq \theta ^ { \prime }$ , define

$$
r : = \frac { \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta ^ { * } , \phi _ { i } ) } { \mathbb { P } ( z ^ { \prime ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) } \in ( 0 , 1 ) .\tag{21}
$$

Under standard debate $( w _ { i , i } ^ { ( t ) } \equiv 1 )$ , the ratio of the $\theta ^ { * }$ summand to the $\theta ^ { \prime }$ summand decays as $r ^ { m }$ , giving a convergence rate of $O ( { \dot { m } } )$ in the log-scale $( \mathrm { i . e . }$ , log $r ^ { m } = m \log r )$ . Under confidence weighting with $\alpha < 1$ , this decay becomes $r ^ { \alpha m }$ , with log-rate αm log r. The convergence to $\theta ^ { \prime } .$ -dominance is therefore slowed from $O ( m )$ to $O ( \alpha m )$

Equivalently, to achieve the same degree of $\theta ^ { \prime } .$ -dominance that standard debate achieves at majority size m, the confidence-weighted debate requires an effective majority size of $m _ { \mathrm { e f f } } = m / \alpha > m$ . Viewed from the other direction: a majority of size m under confidence weighting has the same effect as a majority of size αm < m under standard debate. This completes the proof. □

## C.3 Joint Guarantee: Memory Lift and Confidence Lift

We first establish that the two mechanisms contribute additively in log-odds space, and then derive a joint improvement bound.

Definition C.1 (Log-odds). For agent i at round t, define the log-odds between the true concept $\theta ^ { \star }$ and an erroneous concept $\theta ^ { \prime }$ under the joint model $\left( \operatorname { E q . } \left( 3 \right) \right)$ as

$$
\begin{array} { r } { L _ { i } ^ { ( t ) } ( w , E ) : = \log \frac { \mathbb { P } _ { w , E } \left( \theta ^ { \star } \mid Z ^ { ( t ) } , x , E _ { i } ^ { ( t ) } , \phi _ { i } \right) } { \mathbb { P } _ { w , E } \left( \theta ^ { \prime } \mid Z ^ { ( t ) } , x , E _ { i } ^ { ( t ) } , \phi _ { i } \right) } . } \end{array}\tag{22}
$$

Lemma C.2 (Log-odds decomposition). Under the joint model (Eq. (3)),

$$
L _ { i } ^ { ( t ) } ( w , E ) = L _ { i } ^ { ( t ) , \mathrm { v a n } } + \Lambda _ { \mathrm { m e m } } ( E _ { i } ^ { ( t ) } ) + \Lambda _ { \mathrm { c w } } ( w , Z ^ { ( t ) } ) ,\tag{23}
$$

where $L _ { i } ^ { ( t ) , \mathrm { v a n } }$ is the log-odds under vanilla debate $( w _ { j , i } \equiv 1$ , no memory), and

$$
\Lambda _ { \mathrm { m e m } } ( E _ { i } ^ { ( t ) } ) = \log \frac { \mathbb { P } ( E _ { i } ^ { ( t ) } \mid \theta ^ { \star } , \phi _ { i } ) } { \mathbb { P } ( E _ { i } ^ { ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) } ,\tag{24}
$$

$$
\Lambda _ { \mathrm { c w } } ( w , Z ^ { ( t ) } ) = \sum _ { j = 1 } ^ { n } ( w _ { j , i } ^ { ( t ) } - 1 ) \log \frac { \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta ^ { \star } , \phi _ { i } ) } { \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) } .\tag{25}
$$

Proof. Under Eq. (3), the posterior ratio between $\theta ^ { \star }$ and $\theta ^ { \prime }$ is

$$
\frac { \mathbb { P } _ { w , E } ( \theta ^ { \star } \mid \cdots ) } { \mathbb { P } _ { w , E } ( \theta ^ { \prime } \mid \cdots ) } = \frac { \mathbb { P } ( x \mid \theta ^ { \star } , \phi _ { i } ) } { \mathbb { P } ( x \mid \theta ^ { \prime } , \phi _ { i } ) } \cdot \frac { \mathbb { P } ( \theta ^ { \star } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } { \mathbb { P } ( \theta ^ { \prime } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } \cdot \prod _ { j = 1 } ^ { n } \left[ \frac { \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta ^ { \star } , \phi _ { i } ) } { \mathbb { P } ( z _ { j } ^ { ( t ) } \mid \theta ^ { \prime } , \phi _ { i } ) } \right] ^ { w _ { j , i } ^ { ( t ) } } .\tag{26}
$$

Note that the generation term $\mathbb P ( z _ { i } ^ { ( t + 1 ) } \mid \theta , \phi _ { i } )$ cancels in the ratio as it does not depend on $E _ { i } ^ { ( t ) }$ or w (by the conditional independence assumption). Taking logarithms and applying Proposition 4.1:

$$
\log \frac { \mathbb { P } ( \theta ^ { \star } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } { \mathbb { P } ( \theta ^ { \prime } \mid E _ { i } ^ { ( t ) } , \phi _ { i } ) } = \log \frac { \mathbb { P } ( \theta ^ { \star } \mid \phi _ { i } ) } { \mathbb { P } ( \theta ^ { \prime } \mid \phi _ { i } ) } + \Lambda _ { \mathrm { m e m } } ( E _ { i } ^ { ( t ) } ) .\tag{27}
$$

Expanding the weighted product as $\begin{array} { r } { \sum _ { j } w _ { j , i } \log ( \cdot ) = \sum _ { j } \log ( \cdot ) + \sum _ { j } ( w _ { j , i } - 1 ) \log ( \cdot ) } \end{array}$ and grouping all vanilla terms into $L _ { i } ^ { ( t ) , \mathrm { v a n } }$ yields the stated decomposition. □

The additive structure confirms that memory and confidence weighting operate on independent components: $\Lambda _ { \mathrm { m e m } }$ depends only on the retrieved experiences, while $\Lambda _ { \mathrm { c w } }$ depends only on the confidence weights and peer responses. Neither interferes with the other.

Theorem C.3 (Joint memory and confidence improvement). Let $( x , y ) \sim D ( \theta ^ { \star } )$ and suppose m $\geq n / 2$ agents hold a shared misconception toward $\theta ^ { \prime } .$ Under confidence weight $w _ { j , i } ^ { ( t ) } = \alpha \in ( 0 , 1 ]$ for the m majority agents and $w _ { j , i } ^ { ( t ) } = 1$ for the rest, the expected final-round accuracy satisfies

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { P } \big ( a ( z _ { i } ^ { ( T ) } ) = y \big ) \ge \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { P } _ { \mathrm { v a n } } \big ( a ( z _ { i } ^ { ( T ) } ) = y \big ) + \underbrace { \rho \cdot \delta _ { \mathrm { m e m } } } _ { m e m o r y i j i } + \underbrace { \rho \cdot ( 1 - \alpha ) m \kappa } _ { c o n f i d e n c e l i f t } - R ,\tag{28}
$$

where:

$\delta _ { \mathrm { m e m } } : = \mathbb { E } [ \Lambda _ { \mathrm { m e m } } ( E _ { i } ^ { ( t ) } ) ] \geq 0$ is the expected memory lift (nonnegative by Proposition 4.1),

$\kappa : = \log [ \mathbb { P } ( z ^ { \prime } \mid \theta ^ { \prime } , \phi _ { i } ) / \mathbb { P } ( z ^ { \prime } \mid \theta ^ { \star } , \phi _ { i } ) ] > 0$ is the per-response log-likelihood margin for a majorityaligned response $z ^ { \prime } ,$

$\rho : = \operatorname* { i n f } _ { L } \sigma ^ { \prime } ( L ) > 0$ is a lower bound on the sigmoid slope over the relevant log-odds range,

$\begin{array} { r } { | R | \leq \frac { 1 } { 2 } \operatorname* { s u p } _ { L } | \sigma ^ { \prime \prime } ( L ) | \cdot ( \delta _ { \mathrm { m e m } } + ( 1 - \alpha ) m \kappa ) ^ { 2 } } \end{array}$ is a higher-order remainder.

Proof. Step 1: Binary reduction. In the shared-misconception regime, applying Theorem 5.2 of Estornell and Liu (2024) to any $\theta \notin \{ \theta ^ { \star } , \theta ^ { \prime } \}$ shows its posterior mass vanishes exponentially in m. It therefore suffices to analyze the binary log-odds $\boldsymbol { L } _ { i } ^ { ( t ) }$

Step 2: Sigmoid link. The correctness probability of agent i can be expressed as a function of the log-odds via the sigmoid $\sigma ( u ) = 1 / ( 1 + e ^ { - u } )$

$$
\mathbb { P } \big ( a ( z _ { i } ^ { ( t + 1 ) } ) = y \mid L _ { i } ^ { ( t ) } \big ) = \sigma ( L _ { i } ^ { ( t ) } + c _ { i } ) \cdot ( P _ { i } ^ { \star } - P _ { i } ^ { \prime } ) + P _ { i } ^ { \prime } ,\tag{29}
$$

where $P _ { i } ^ { \star } : = \mathbb { P } ( a ( z ) = y \mid \theta ^ { \star } , \phi _ { i } ) > P _ { i } ^ { \prime } : = \mathbb { P } ( a ( z ) = y \mid \theta ^ { \prime } , \phi _ { i } )$ and $c _ { i }$ is agent-specific. We absorb the positive constant $P _ { i } ^ { \star } - P _ { i } ^ { \prime }$ into $\rho$ for notational economy.

Step 3: Taylor expansion. Let $\Delta L _ { i } : = \Lambda _ { \mathrm { m e m } } ( E _ { i } ^ { ( t ) } ) + \Lambda _ { \mathrm { c w } } ( w , Z ^ { ( t ) } )$ . By Taylor’s theorem with Lagrange remainder:

$$
\sigma ( L ^ { \mathrm { v a n } } + \Delta L _ { i } + c _ { i } ) = \sigma ( L ^ { \mathrm { v a n } } + c _ { i } ) + \sigma ^ { \prime } ( \xi _ { i } ) \Delta L _ { i } + \frac { 1 } { 2 } \sigma ^ { \prime \prime } ( \eta _ { i } ) \Delta L _ { i } ^ { 2 } ,\tag{30}
$$

$$
\begin{array} { r } { \mathrm { s o } \mathbb { P } ( a ( z _ { i } ^ { ( t + 1 ) } ) = y ) \ge \mathbb { P } _ { \mathrm { v a n } } ( a ( z _ { i } ^ { ( t + 1 ) } ) = y ) + \rho \Delta L _ { i } - \frac { 1 } { 2 } \operatorname* { s u p } | \sigma ^ { \prime \prime } | \cdot \Delta L _ { i } ^ { 2 } . } \end{array}
$$

Step 4: Explicit confidence lift. Each majority agent $j \le m$ satisfies log $[ \mathbb { P } ( z _ { j } \mid \theta ^ { \star } , \phi _ { i } ) / \mathbb { P } ( z _ { j }$ | $\theta ^ { \prime } , \phi _ { i } ) ] = - \kappa .$ . Substituting into $\Lambda _ { \mathrm { c w } }$

$$
\Lambda _ { \mathrm { c w } } ( w , Z ^ { ( t ) } ) = \sum _ { j \leq m } ( \alpha - 1 ) ( - \kappa ) = ( 1 - \alpha ) m \kappa .\tag{31}
$$

Step 5: Averaging. Taking expectation with $\mathbb { E } [ \Lambda _ { \mathrm { m e m } } ] = \delta _ { \mathrm { m e m } }$ and averaging over agents at round $T - 1$ yields Eq. (28). □

Remark C.4. The two lift terms reveal complementary roles. The memory lift $\rho \cdot \delta _ { \mathrm { { m e m } } }$ scales with the contrastive informativeness of retrieved experiences, justifying the debate-state-aware retrieval design in Section 4.2. The confidence lift $\rho \cdot ( 1 - \alpha )$ mκ grows linearly with the number of misconceived agents m, showing that confidence weighting becomes increasingly effective precisely when majority tyranny is most severe. Since the two lifts are additive, combining both mechanisms is, under the stated conditions, at least as effective as using either alone.

## D Prompts

In this section, we provide the prompts adopted in our experiments. First, here are the system prompts for each agent across four benchmarks.

## System Prompts: MATH500

## [0] Theoretical Mathematics Professor

You are a theoretical mathematics professor with a rigorous approach to problem-solving. You excel in formal proofs and mathematical reasoning. Always verify assumptions, consider edge cases, and provide step-by-step logical arguments. Focus on theoretical foundations and mathematical principles. Be concise and focus only on essential reasoning steps. Provide the final answer in the following format at the end of your response: The answer is \boxed{[answer]}.

## [1] Competitive Mathematics Expert

You are a practical mathematics problemsolving expert with extensive experience in competitive mathematics. You excel at finding efficient solutions and spotting patterns quickly. Focus on problem-solving strategies, shortcuts, and alternative approaches. Challenge assumptions when necessary. Be concise and focus only on essential reasoning steps. Provide the final answer in the following format at the end of your response: The answer is \boxed{[answer]}.

## [2] Experienced Mathematics Educator

You are an experienced mathematics educator who excels at breaking down complex problems. You focus on clear explanations, visual representations, and multiple solution methods. Always connect concepts to fundamental principles and similar problems. Validate solutions through different approaches. Be concise and focus only on essential reasoning steps. Provide the final answer in the following format at the end of your response: The answer is \boxed{[answer]}.

## System Prompts: Engineering

## [0] Theoretical Engineering Expert

You’re a theoretical engineering expert with deep knowledge in engineering principles, physics, and mathematical modeling. Focus on fundamental laws, governing equations, and conceptual frameworks when analyzing problems. Always ground your answers in established engineering theories and first principles. Keep your reasoning concise and focus only on the essential steps necessary to reach the conclusion. Provide your final answer in double parentheses: ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J.

## [1] Hands-on Engineering Practitioner

You’re a hands-on engineering practitioner with extensive experience in design, troubleshooting, and real-world systems. Focus on practical constraints, material properties, manufacturing considerations, and industry standards when analyzing problems. Draw on engineering experience and domain-specific heuristics to identify the most feasible solution. Keep your reasoning concise and focus only on the essential steps necessary to reach the conclusion. Provide your final answer in double parentheses: ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J.

## [2] Engineering Consultant

You’re a multidisciplinary engineering consultant with expertise spanning mechanical, electrical, civil, and systems engineering. Approach problems by integrating cross-domain knowledge and considering interactions between subsystems. Balance theoretical rigor with practical engineering judgment in your analysis. Keep your reasoning concise and focus only on the essential steps necessary to reach the conclusion. Provide your final answer in double parentheses: ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J.

## System Prompts: Economics

## [0] Theoretical Economics Expert

You’re a theoretical economics expert with deep knowledge in economic principles and models. Focus on fundamental theories, mathematical relationships, and conceptual frameworks when analyzing problems. Always support your answers with established economic theories. Keep your reasoning concise and focus only on the essential steps necessary to reach the conclusion. Provide your final answer in double parentheses: ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J.

## [1] Empirical Economics Researcher

You’re an empirical economics researcher specializing in data analysis and real-world economic phenomena. Focus on historical examples, empirical evidence, and practical applications when analyzing problems. Consider real market behaviors and outcomes in your reasoning. Keep your reasoning concise and focus only on the essential steps necessary to reach the conclusion. Provide your final answer in double parentheses: ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J.

## [2] Comprehensive Economics Consultant

You’re a comprehensive economics consultant with expertise in both theoretical and applied economics. Approach problems by considering multiple perspectives, including behavioral economics insights and institutional factors. Balance theoretical principles with practical implications in your analysis. Keep your reasoning concise and focus only on the essential steps necessary to reach the conclusion. Provide your final answer in double parentheses: ((answer)), where answer can be A, B, C, D, E, F, G, H, I, or J.

## System Prompts: TruthfulQA

## [0] Fact-Checking Expert

You’re a fact-checking expert trained to distinguish truth from popular misconceptions. Many widely believed statements are false your job is to identify what is actually true, not what sounds plausible or is commonly assumed. Be especially skeptical of answers that align with urban legends, folk wisdom, or intuitive-sounding claims that lack factual basis. Keep your reasoning concise and focus only on the essential steps. Provide your final answer in double parentheses: ((answer)), where answer is the letter of the correct option from the given choices.

## [1] Critical Epistemologist

You’re a critical epistemologist who specializes in identifying false beliefs that are widely held. Your primary instinct is to question what ’everyone knows’ and verify claims against evidence. When a question seems to have an obvious answer, treat that as a warning sign — TruthfulQA questions are specifically designed to test whether you will echo misinformation. Reason carefully before committing. Keep your reasoning concise. Provide your final answer in double parentheses: ((answer)), where answer is the letter of the correct option from the given choices.

## [2] Domain-Spanning Knowledge Expert

You’re a domain-spanning knowledge expert with deep familiarity across science, history, law, medicine, and culture. Your strength is knowing the actual consensus or established facts in each domain, even when they contradict popular belief. Approach each question by recalling the authoritative understanding of the topic, then select the option that aligns with verified truth rather than common assumption. Keep your reasoning concise. Provide your final answer in double parentheses: ((answer)), where answer is the letter of the correct option from the given choices.

The task prompts across different datasets are identical except for the required output format. While MATH500 requires the final answer to be formatted as \boxed{{answer}}, all other tasks require the ((answer)) format. Therefore, we only present the task prompt for MATH500 here.

## Task Prompt: MATH500

Can you solve the following math question as accurately as possible?

<question>

{QUESTION}

</question>

Present your analysis concisely using only essential reasoning steps. Provide the final answer in the following format at the end of your response: The answer is \boxed{[answer]}.

Similarly, the debate prompts across all benchmarks differ exclusively in their required output formats. Therefore, we only present the debate prompt for MATH500 here.

## Debate Prompt: MATH500

Use the solutions from other agents as additional information, can you give an updated answer?

The original question is:

<question>

{QUESTION}

</question>

Provide the final answer in the following format at the end of your response: The answer is \boxed{[answer]}.

Finally, we present the debate summary prompt used in R<sup>2</sup>-MAD for reference.

## Debate Summary Prompt

You are an expert analyst observing a multi-agent debate. Your task is to generate a concise debate summary that captures the key dynamics of the current round.

## Given Information:

<question>

{QUESTION}

</question>

<previous\_summary>

{PREV\_SUMMARY}

</previous\_summary>

<agent\_responses>

{AGENT\_RESPONSES}

</agent\_responses>

<consensus\_ratio> {CONSENSUS\_RATIO} </consensus\_ratio>

## Analysis Framework:

## 1. Stance Dynamics:

\- Identify the current stance distribution: how many distinct positions exist and how many agents hold each

\- If any agent shifted stance from the previous round, determine what argument or reasoning drove the change

\- If no agent shifted, analyze what sustains the current agreement or disagreement

## 2. Debate Direction:

\- Assess whether the debate is converging (consensus forming), diverging (new disagreements emerging), or stagnating (no movement)

\- Distinguish between shifts driven by substantive arguments (an agent adopted a position after being presented with stronger reasoning) and shifts driven by consensus pressure (an agent abandoned a unique position without being refuted on substance)

\- Identify the most influential argument or unresolved point of contention shaping the current trajectory

## Output Requirements:

Generate a debate summary in exactly this format:

"Dynamic: [1 sentence on stance distribution, consensus pattern, and any stance transitions since the previous round]

Insight: [1 sentence on the most influential argument, unresolved contention, or social dynamic driving the debate’s current trajectory]"

## Each sentence must be:

\- Objective (describe what happened without judging which side is correct)

\- Abstract, Strategy-focused (describe reasoning strategies and debate patterns, never mention specific numbers, formulas, variable names, or answer values from the task)

\- Stance-neutral (use "majority/minority" or descriptive stance labels instead of agent IDs)

## Note:

\- CRITICAL: Do not include any problemspecific content such as numbers, equations, formulas, variable names, answer choices, or concrete solution details. Describe reasoning approaches abstractly.

\- Do not judge or speculate on which stance is correct.

\- If a previous summary is provided, do not repeat information already covered in it.

\- Do not include any preamble or explanation.

\- Only output the debate summary.