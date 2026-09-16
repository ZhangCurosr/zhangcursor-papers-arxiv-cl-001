# ThinkFlow: Self-Evolving Probabilistic Latent Memory for Lifelong Conversational Agents

Cai Ke<sup>1,2</sup>, Xin Liu<sup>1∗</sup>, Han Zhang<sup>1</sup>, Jiangyue Yan<sup>1,2</sup>, Zike Yuan<sup>1,2</sup>, Ling Deng<sup>3</sup>, Yue Yu<sup>1</sup>, Hui Wang<sup>1</sup>, and Ruifeng Xu<sup>2,1∗</sup>

<sup>1</sup>Pengcheng Laboratory, China <sup>2</sup>Harbin Institute of Technology, Shenzhen, China <sup>3</sup>China Unicom Greater Bay Area Innovation Institute, China kecai@stu.hit.edu.cn, xuruifeng@hit.edu.cn

## Abstract

Lifelong conversational agents rely on memory systems to maintain deep, context-aware interactions with users. However, existing explicit textual memory pipelines suffer from a severe information bottleneck, often losing subtle behavioral patterns and emotional shifts. Furthermore, being typically static post-deployment, they cannot autonomously adapt to personal habits and preferences without manual feedback. Cognitive science, however, suggests that humans maintain mental models purely in a latent space and continuously refine them through predictive coding. Inspired by this, we propose ThinkFlow, a novel end-to-end latent memory framework for lifelong conversational agents. ThinkFlow bypasses the text bottleneck by dynamically compressing conversational flows into probabilistic latent memory skills, autonomously consolidating complex user states into disentangled, continuous vectors without semantic interference. To break this barrier, we introduce a test-time evolution paradigm. By coupling teacher-guided latent alignment to bootstrap the initial state with a self-supervised next-user-utterance prediction task for continuous refinement, the framework successfully overcomes cold-start challenges and achieves label-free lifelong personalization. Extensive experiments on long-term conversation benchmarks demonstrate that ThinkFlow significantly outperforms prevailing memory systems, providing highly personalized and contextually accurate responses over extended multi-session interactions.

## 1 Introduction

Recent advances in large language models (LLMs) have transformed them from static question-answering tools into conversational agents expected to build engaging, lifelong relationships with users (Xu et al., 2022; Bae et al., 2022; Zhang et al., 2023; Lu et al., 2023; Jang et al., 2024; Zhong et al., 2024; Li et al., 2025a; Liang et al., 2026; Ke et al., 2026a). This evolution has catalyzed a paradigm shift toward personalized memory retrieval (Ong et al., 2025; Tan et al., 2025; Zhang et al., 2025; Jiang et al., 2025a,b; Zhang et al., 2026b), which focuses on capturing user-specific histories, evolving preferences, and highly contextualized interactions to tailor responses to a particular user at a specific moment. However, in real-world scenarios, standard turnby-turn interactions inevitably suffer from episodic amnesia, failing to satisfy the demand for deep, persistent personalization over prolonged multi-session engagements (Maharana et al., 2024; Wu et al., 2025). Simply feeding the entire raw multi-session dialog history directly into the context window is not only computationally prohibitive but also severely exacerbates the semantic dilution problem, leading to retrieval degradation (Liu et al., 2024a; Xu et al., 2025a; Laban et al., 2026). This underscores the critical need for an efficient, dynamic memory mechanism capable of managing lifelong human-computer companionship while maintaining scalable personalization.

![](images/b13f205b401a1840645fda03efe87e803f3b7df04e33e3aa4bd36f3206afa954.jpg)  
Figure 1: Conceptual overview of our ThinkFlow, mitigating the text bottleneck (e.g., Keyword Noise) of explicit memory via dynamic latent memory skills.

The current paradigm primarily relies on explicit textbased memory systems, utilizing complex pipelines to manage the entire memory lifecycle from the construction and management to the retrieval of historical conversations. While yielding remarkable results, these explicit methods fundamentally suffer from a severe information bottleneck (Packer et al., 2023; Liu et al., 2024b; Mei et al., 2024; Wang et al., 2025a; Chhikara et al., 2025; Xu et al., 2025b; Kang et al., 2025; Fang et al., 2026). By forcing the rich, continuous flow of human interaction to be materialized into rigid, discrete plain text, they inherently strip away subtle nuances including implicit user preferences, shifting emotional states, and unspoken behavioral patterns. Furthermore, these text-based memory banks are typically static postdeployment and lack the ability to autonomously adapt to user idiosyncrasies without explicit feedback. This limitation prevents LLMs from developing a deep, seamlessly evolving understanding of the user, keeping them from providing truly empathetic and human-like lifelong companionship. This points towards the need for a memory mechanism that, much like human cognition, operates implicitly and continuously self-evolves without relying on explicit text generation.

According to Cognitive Science, specifically the Implicit Theory of Mind (Apperly and Butterfill, 2009; Schneider et al., 2012), humans do not memorize discrete text summaries of past interactions; rather, they naturally and unconsciously track others’ mental states, continuously updating their internal mental models purely in a latent, abstract space. Furthermore, based on Predictive Coding (Clark, 2013; Millidge et al., 2021), the human brain constantly generates top-down predictions of future sensory inputs and refines its internal models based on prediction errors. As illustrated in Figure 1 (a), traditional explicit memory systems suffer from the text bottleneck, where complex behavioral nuances are lost during summarization, leading to rigid and static profiles. In contrast, as shown in Figure 1 (b), our proposed latent memory dynamically compresses the continuous conversational flow into disentangled skill vectors (e.g., Facts, Emotions, Preferences). Driven by implicit feedback from predicting the user’s next action, these latent skills autonomously update and evolve. Therefore, we argue that by abandoning the explicit text bottleneck and maintaining a continuously evolving memory purely in the latent space through selfsupervised predictive feedback, we can achieve true, zero-shot personalization in long-term conversations.

To reach this goal, we propose ThinkFlow, an end-to-end latent memory framework designed to autonomously extract, filter, and align historical information. Specifically, to overcome the text bottleneck, we introduce Probabilistic Latent Memory Skills (PLMS), which compress redundant context into K dense, disentangled skill vectors that act as specialized cognitive receptors. To mitigate semantic dilution and filter out conversational noise, a Gated Latent Consolidator (GLC) strictly controls the updating of these skills. Subsequently, a Context-Aware Hyper-Aligner (CAHA) dynamically bridges the distributional gap between historical latent skills and the current semantic context of the LLM. Most importantly, to break the deployment bottleneck of static models, we introduce a Self-Supervised Test-Time Evolution paradigm. After an initial Teacher-Guided Latent Alignment, ThinkFlow engages in continuous lifelong learning via a Next-User-Utterance Prediction task. It utilizes the user’s implicit reaction as an abundant, free supervisory signal to refine its latent skills based on prediction errors. Experimental results on long-term conversation benchmarks demonstrate that ThinkFlow significantly enhances personalized response generation and memory utilization. The contributions of this work can be summarized as follows:

• We explore a new end-to-end latent memory paradigm for lifelong conversational agents, bypassing the traditional explicit text bottleneck to effectively prevent information loss during longterm interactions.

• We are the first to dynamically compress the continuous conversational flow into disentangled skill vectors, enabling the agent to autonomously filter, extract, and consolidate complex user states without semantic interference.

• We introduce a novel self-supervised test-time evolution paradigm. By leveraging implicit predictive feedback during interactions, the proposed framework effectively overcomes the initial cold-start hurdle and achieves continuous, label-free lifelong personalization.

• Experimental results demonstrate the superiority of ThinkFlow over prevailing memory systems, showcasing its ability to provide highly personalized and contextually accurate responses in extended multi-session interactions.

## 2 Related Work

Explicit Memory Paradigm. Long-term memory is critical for maintaining personalization and coherence in lifelong conversational agents. Existing systems predominantly rely on explicit textual memory, which often starts with structured retrieval-augmented generation to organize past conversations into structured topologies like hierarchical trees or knowledge graphs for contextaware reasoning (Sarthi et al., 2024; Edge et al., 2024; Rezazadeh et al., 2025). Moving beyond static structures, memory-augmented generation paradigms dynamically extract discrete facts or generate condensed textual summaries of previous sessions to build plug-andplay memory banks (Lu et al., 2023; Zhong et al., 2024; Chen et al., 2025; Li et al., 2025a; Wang et al., 2025b; Ke et al., 2025; Ong et al., 2025; Fang et al., 2026). To actively govern these records, modern frameworks transition to agentic memory management, employing autonomous agents or hierarchical storage architectures to read, write, and update these explicit textual blocks (Packer et al., 2023; Xu et al., 2025b; Chhikara et al., 2025; Kang et al., 2025; Ke et al., 2026a,b). Different from these methods that rely on explicit textual pipelines and suffer from severe information loss or redundancy, our ThinkFlow bypasses this text bottleneck entirely by compressing and evolving conversational flows purely in the latent space as continuous, probabilistic memory skills.

Latent Memory Paradigm. To mitigate the token costs and information bottlenecks of explicit textual memory, latent space has emerged as a promising memory substrate due to its token efficiency, machine-native representation, and end-to-end learnability (Hu et al., 2025; Yu et al., 2026). For textual agents, pioneer works encode multi-session experiences into continuous vector spaces to support agentic reasoning, such as representing historical sessions as continuous trajectory embeddings (Zhang et al., 2026c) or caching key contextual patterns into latent memories (Xu et al., 2025c; Li et al., 2025b; Liu et al., 2025; Zhang et al., 2026a). Different from existing latent memory methods that rely on static latent vectors, ThinkFlow models memory as probabilistic memory skills to capture interaction uncertainty. Furthermore, our method continuously predicts the user’s next response to capture their implicit feedback and shifting preferences, achieving self-evolution during online conversations and enabling label-free lifelong personalization.

## 3 Methodology

Drawing inspiration from the Implicit Theory ofMind in cognitive science, humans naturally and unconsciously track others’ mental states, selectively consolidating salient interaction histories to continuously update their internal mental models purely in the latent space. Inspired by this, to address the information loss caused by the traditional text-based memory bottleneck, we propose ThinkFlow. Instead of saving chat history as discrete text, ThinkFlow operates purely in the latent space by compressing the continuous thinking flow into probabilistic memory skills. For a conversation at turn t, let $x _ { t }$ be the user query, $a _ { t }$ be the agent response, $H _ { < t }$ be the text history, and $M _ { t } \in \mathbb { R } ^ { K \times \check { D } }$ be the latent memory state comprising K skill slots and D dimensions.

## 3.1 Overall Architecture

At the core of this architecture is the concept of latent memory skills. Instead of a monolithic hidden state, we decompose the memory into multiple continuous skill vectors. This skill-centric design is crucial, as it allows the system to autonomously capture and decouple diverse aspects of the user’s profile—such as factual events, emotional states, and implicit preferences—into specialized channels. The architecture comprises three interconnected components.

Probabilistic Latent Memory Skills. To extract finegrained semantics from the redundant conversational context, the PLMS compresses the thought stream (hidden states) $E _ { t } \in \mathbb { R } ^ { L \times D }$ of the current turn into K dense memory skills. These skills act as specialized cognitive receptors. To capture the intricate dependencies between the predefined memory slots and the input sequence, we apply cross-attention, where a set of learnable latent memory skills $S \in \mathbb { R } ^ { K \times D }$ attends to $E _ { t }$ This skill-based design is crucial for long-term agents: it enforces a disentangled memory skill representation where different skills autonomously specialize in distinct facets of the user profile, such as factual events, personal preferences, or emotional states. By doing so, it prevents semantic interference among diverse conversational topics and allows the agent to precisely retrieve specific memory skill fragments when needed.

To prevent the variance of the compressed skills from collapsing into a deterministic point mass, we introduce a distribution-preserving scaling factor. The scaled attention output is defined as:

$$
C _ { t } = \Big ( \mathrm { S o f t m a x } \big ( \frac { S E _ { t } ^ { T } } { \sqrt { D } } \big ) E _ { t } \Big ) \times \frac { 1 } { \sqrt { c } }\tag{1}
$$

where $c = L / K$ is the compression rate. To explicitly model the epistemic uncertainty of the extracted memory skills, the PLMS outputs a mean vector $\mu _ { t }$ and a log-variance vector log $\sigma _ { t } ^ { \dot { 2 } }$ through two parallel linear layers applied to $C _ { t }$ . During training, to enable gradient backpropagation through the stochastic sampling process, we employ the reparameterization trick:

$$
\hat { S } _ { n e w } = \mu _ { t } + \sigma _ { t } \odot \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I )\tag{2}
$$

To regularize the distribution, we apply a Kullback-Leibler (KL) divergence loss $\mathcal { L } _ { K I }$ to push the approximate posterior towards a standard Gaussian prior.

Gated Latent Consolidator. To mitigate the semantic dilution problem and filter out useless chat noise (e.g., redundant greetings), the GLC strictly controls how much new information should be permanently stored. To explicitly quantify the novelty of incoming information against the established memory skills, we introduce an information gain gate $g _ { t } \in ( 0 , \dot { 1 } ) ^ { K \times D }$ . This gate is learned via a multi-layer perceptron operating on the concatenated representations of the new and old memory skills:

$$
g _ { t } = \sigma \Big ( W _ { 2 } \mathrm { R e L U } \big ( W _ { 1 } [ \hat { S } _ { n e w } \parallel S _ { o l d } ] \big ) \Big )\tag{3}
$$

To maintain the temporal sequence of the latent flow while gracefully forgetting outdated elements, we utilize a Gated Recurrent Unit (GRU) core. The latent memory skills are conditionally updated step-by-step:

$$
S _ { t } = g _ { t } \odot \mathrm { G R U } ( \hat { S } _ { n e w } , S _ { o l d } ) + ( 1 - g _ { t } ) \odot S _ { t - 1 }\tag{4}
$$

![](images/99692bbd78a90fdd99065a47f8ce78d2bb7c32c40e3554d8c56263f4f1d84f7e.jpg)  
Figure 2: Illustration of our ThinkFlow (Left) and training strategy (Right). The architecture is an end-to-end memory system designed to extract, filter, and align historical information without relying on explicit text generation.

Context-Aware Hyper-Aligner. To bridge the distributional gap between historical latent skill representations and the current semantic context of the language model, the CAHA translates historical memory skills dynamically. To achieve this fine-grained adaptation without adding massive parameters, we adopt a hypernetwork approach. It generates a low-rank transformation matrix $W _ { a d a p t } \in \mathbb { R } ^ { D \times D }$ based on the linguistic features of the current user query x<sub>t</sub>:

$$
W _ { a d a p t } = W _ { d o w n } ( x _ { t } ) \times W _ { u p } ( x _ { t } )\tag{5}
$$

where $W _ { d o w n } \in \mathbb { R } ^ { D \times { r } }$ and $W _ { u p } \in \mathbb { R } ^ { r \times D }$ form a bottleneck of rank r. To compute the final aligned memory skills, we perform a residual projection: $S _ { a l i g n e d } =$ $S _ { t } + S _ { t } \times W _ { a d a p t }$ . Finally, to condition the generation process, these aligned skills are prepended to the language model as soft prompts.

## 3.2 Self-Supervised Test-Time Evolution

Traditional memory models suffer from a severe deployment bottleneck: they rely heavily on static, humanannotated dialogue datasets, making them incapable of adapting to individual user idiosyncrasies after deployment. To achieve true personalization without requiring manual labels, ThinkFlow introduces a Self-Supervised Test-Time Evolution paradigm. This two-phase training strategy allows the agent to continuously refine its latent memory skills purely from real-world interactions.

Phase 1: Teacher-Guided Latent Alignment. To address the cold-start problem during a user’s first session, we conduct a Teacher-Guided Latent Alignment. To transfer the comprehensive contextual understanding from a computationally expensive text-based teacher to our efficient latent memory skill student, the powerful teacher model reads the full plaintext history $H _ { < t }$ and outputs a target probability distribution $P _ { T } ( y \mid H _ { < t } , x _ { t } )$ . The student model only reads the current query $x _ { t }$ and the latent memory skills $S _ { t } ,$ outputting a predictive distribution $\textstyle P _ { S } ( y \mid S _ { t } , x _ { t } )$ . To force the student to mimic the teacher and establish the initial memory skill anchor, we minimize objective:

$$
\begin{array} { r l } {  { \mathcal { L } _ { P h a s e 1 } = \mathcal { D } _ { K L } \Big ( P _ { T } ( y \mid H _ { < t } , x _ { t } ) \mid \mid P _ { S } ( y \mid S _ { t } , x _ { t } ) \Big ) } \quad } & { { } } \\ { + \alpha \mathcal { L } _ { K L } } & { { } } \end{array}\tag{6}
$$

Phase 2: Next-User-Utterance Prediction. To enable continuous lifelong learning without manual annotations, the teacher model goes offline from Session 2 onwards. Drawing from Predictive Coding in cognitive science, human brains constantly generate top-down predictions of future sensory inputs and update their internal models based on prediction errors. Inspired by this, we cast the memory skill evolution problem as a Next-User-Utterance Prediction task, utilizing the user’s implicit feedback as an abundant and free supervisory signal. At turn t, the agent generates response $a _ { t }$ and updates the memory skills to $S _ { t + 1 }$ . At turn $t + 1$ , the user provides a new query $x _ { t + 1 }$ . To construct a pure self-supervised signal that grounds the memory skill evaluation on real-world predictive success, the system uses the latent memory skills $S _ { t + 1 }$ and its previous response $a _ { t }$ to predict the user’s next utterance $x _ { t + 1 }$

If the memory skills $S _ { t + 1 }$ fail to capture the core context, the model will fail to predict the user’s reaction, resulting in a high cross-entropy loss $\mathcal { L } _ { C E }$ . To shield the memory skill bank from being polluted by bad updates, this high loss directly forces the GLC gate to close. Furthermore, to encourage the model to maintain sharp, confident predictions and prevent ambiguous semantic collapse, we introduce a minimum entropy confidence prior $\mathcal { L } _ { C o n f }$ . The loss for Phase 2 is formulated as:

$$
\begin{array} { r l } { \mathcal { L } _ { P h a s e 2 } = \mathcal { L } _ { C E } ( x _ { t + 1 } \mid S _ { t + 1 } , a _ { t } ) } & { { } } \\ { + \beta \mathcal { L } _ { C o n f } + \alpha \mathcal { L } _ { K L } } & { { } } \end{array}\tag{7}
$$

## 4 Experiments

## 4.1 Experimental Settings

Datasets and Benchmarks. Following (Ong et al., 2025), we evaluate our method on three long-term opendomain conversation datasets: Multi-Session Chat (MSC) (Xu et al., 2022), Conversation Chronicles (CC), (Jang et al., 2023), and GapChat (GC) (Zhang et al., 2023) and a personalized memory question answering benchmark: PersonaMem (Jiang et al., 2025a). More details are shown in Appendix A.

Models and Baselines. For backbone, we evaluate on two open-source long-context LLMs: 1) Llama-3.2 3B-Instruct (Grattafiori et al., 2024). 2) Qwen3- 8B (Yang et al., 2025). We compare our ThinkFlow against various baselines. 1) Long Context: which use all the conversation histories. 2) Structured Retrieval-Augmented Generation: GraphRAG (Edge et al., 2024) and MemTree (Rezazadeh et al., 2025). 3) Agentic Memory Management: MemGPT (Packer et al., 2023), A-Mem (Xu et al., 2025b), Mem0 (Chhikara et al., 2025), and MemoryOS (Ong et al., 2025). 4) Memory-Augmented Generation: MemoChat (Lu et al., 2023), MemoryBank (Zhong et al., 2024), Rsum (Wang et al., 2025b), COMEDY (Chen et al., 2025), LD-Agent (Li et al., 2025a), THEANINE (Ong et al., 2025), and LightMem (Fang et al., 2026). More details are shown in Appendix B.

Evaluation Metrics. We comprehensively evaluate our method on automatic metrics and accuracy: 1) Automatic Metrics. On CC, MSC and GC, we use BLEU-4 (Papineni et al., 2002), ROUGE-L (Lin, 2004), BertScore (Zhang et al., 2019), and Mauve (Pillutla et al., 2021) to automatically evaluate turn-by-turn response generation. 2) Accuracy. On PersonaMem, performance evaluation is based on the accuracy of the generated responses, specifically the proportion of responses that correctly match the user’s current persona and the conversational context.

Implementation Details. To maintain computational efficiency, we employ LoRA (Hu et al., 2022) on LLMs.

For the core parameter, the default number of probabilistic latent memory skills is set to $K = 1 0$ . To ensure strict alignment in the representation space and vocabulary during context distillation, both the Teacher and Student LLMs are instantiated from the same base LLMs. More details are shown in Appendix C.

## 4.2 Performance in Long-Term Generation

Bypassing the explicit text bottleneck via latent memory is crucial for preventing information loss in continuous dialogues. Table 1 presents the evaluation results for turn-by-turn generation. Directly feeding the full dialogue history into the long context yields sub-optimal performance, highlighting the severe issue of semantic dilution. Furthermore, while various explicit memory paradigms improve performance over the long-context baseline, their gains are fundamentally constrained by the information bottleneck of discrete text summaries. In contrast, our proposed ThinkFlow framework consistently achieves the best overall performance across all datasets. Notably, ThinkFlow demonstrates massive improvements in the Mauve metric, indicating much more coherent and human-like responses without losing subtle conversational nuances.

## 4.3 Performance in Personalized Memory

Tracking the dynamic evolution of user states requires continuous latent updating rather than static text retrieval. Table 2 shows that ThinkFlow consistently achieves the highest average accuracy among comparable methods on the PersonaMem benchmark across extremely long contexts (up to 1M tokens), particularly excelling in complex, dynamic scenarios like tracking user evolution. Remarkably, our ThinkFlow-8B rivals or even surpasses current powerful closed-source and massive open-source (405B) models. These results validate that maintaining memory purely in the latent space effectively overcomes the text bottleneck, accurately capturing user states for lifelong companionship.

## 4.4 Ablation Study

Both structural feature disentanglement and continuous self-supervised evolution are indispensable for maintaining high-quality companionship. Table 3 presents the ablation study of ThinkFlow on the Qwen3-8B backbone. Removing core structural modules (w/o PLMS, GLC, CAHA) leads to substantial performance degradation across datasets, particularly in the Mauve metric. The severe drop observed when removing PLMS validates the fundamental necessity of compressing conversational flow into disentangled skill vectors. Furthermore, omitting the two training phases (w/o Phase1, Phase2) also results in noticeable declines, confirming that both the initial teacher-guided alignment and the continuous test-time evolution are critical for achieving dynamic, label-free personalization.

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Methods</td><td colspan="4">cCC</td><td colspan="4">MSC</td><td colspan="4">GC</td></tr><tr><td>B-4</td><td>R-L</td><td>Bert</td><td>Mauve</td><td>B-4</td><td>R-L</td><td>Bert</td><td>Mauve</td><td>B-4</td><td>R-L</td><td>Bert</td><td>Mauve</td></tr><tr><td></td><td>Long Context (128K)</td><td>0.54</td><td>10.09</td><td>43.12</td><td>35.53</td><td>0.47</td><td>9.94</td><td>44.61</td><td>45.34</td><td>0.20</td><td>3.79</td><td>33.12</td><td>18.85</td></tr><tr><td rowspan="12"></td><td></td><td colspan="7">Explicit Memory Paradigm</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphRAG (2024)</td><td>0.81</td><td>11.37</td><td>45.19</td><td>48.23</td><td>0.62</td><td>10.15</td><td>45.03</td><td>47.61</td><td>0.31</td><td>4.57</td><td>34.11</td><td>19.23</td></tr><tr><td>MemTree (2025)</td><td>0.73</td><td>11.22</td><td>44.57</td><td>51.78</td><td>0.56</td><td>10.02</td><td>44.51</td><td>49.46</td><td>0.28</td><td>4.42</td><td>33.58</td><td>18.65</td></tr><tr><td>MemGPT (2023)</td><td>0.75</td><td>11.53</td><td>40.18</td><td>37.52</td><td>0.52</td><td>9.27</td><td>40.65</td><td>38.81</td><td>0.27</td><td>4.31</td><td>33.17</td><td>18.53</td></tr><tr><td>A-Mem (2025b)</td><td>0.64</td><td>10.35</td><td>44.62</td><td>31.28</td><td>0.47</td><td>10.05</td><td>44.51</td><td>42.48</td><td>0.25</td><td>4.19</td><td>33.43</td><td>17.89</td></tr><tr><td>Mem0 (2025)</td><td>0.67</td><td>10.53</td><td>43.15</td><td>31.49</td><td>0.46</td><td>9.72</td><td>43.24</td><td>44.89</td><td>0.26</td><td>4.23</td><td>33.51</td><td>18.11</td></tr><tr><td>MemoryOS (2025)</td><td>0.69</td><td>10.61</td><td>43.47</td><td>32.17</td><td>0.48</td><td>9.83</td><td>43.69</td><td>45.13</td><td>0.27</td><td>4.37</td><td>33.73</td><td>18.41</td></tr><tr><td>MemoChat (2023)</td><td>0.83</td><td>11.91</td><td>40.73</td><td>38.45</td><td>0.55</td><td>10.11</td><td>41.27</td><td>39.51</td><td>0.26</td><td>4.13</td><td>32.95</td><td>17.93</td></tr><tr><td>Llama3.2-3B MemoryBank (2024)</td><td>1.06</td><td>13.22</td><td>42.10</td><td>44.30</td><td>0.63</td><td>11.27</td><td>42.97</td><td>42.02</td><td>0.33</td><td>5.17</td><td>34.61</td><td>19.87</td></tr><tr><td>Rsum (2025b)</td><td>1.01</td><td>12.87</td><td>41.83</td><td>43.19</td><td>0.61</td><td>11.03</td><td>42.59</td><td>41.77</td><td>0.31</td><td>5.09</td><td>34.27</td><td>19.43</td></tr><tr><td>COMEDY (2025)</td><td>0.61</td><td>9.83</td><td>38.97</td><td>35.21</td><td>0.43</td><td>8.71</td><td>39.81</td><td>36.43</td><td>0.19</td><td>3.51</td><td>31.47</td><td>16.53</td></tr><tr><td>LD-Agent (2025a)</td><td>1.13</td><td>13.51</td><td>42.83</td><td>45.67</td><td>0.67</td><td>11.53</td><td>43.57</td><td>43.11</td><td>0.35</td><td>5.31</td><td>35.13</td><td>20.17</td></tr><tr><td>THEANINE (2025)</td><td>1.09</td><td>13.11</td><td>42.41</td><td>56.73</td><td>0.65</td><td>11.19</td><td>43.19</td><td>51.27</td><td>0.34</td><td>5.23</td><td>34.87</td><td>21.39</td></tr><tr><td>LightMem (2026)</td><td>0.70</td><td>11.68</td><td>45.21</td><td>54.47</td><td>0.65</td><td>11.80</td><td>46.02</td><td>58.38</td><td>0.29</td><td>4.96</td><td>35.04</td><td>19.05</td></tr><tr><td rowspan="7"></td><td></td><td></td><td></td><td></td><td>Latent Memory Paradigm</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SoftCoT (2025c)</td><td>0.25</td><td>3.68</td><td>27.27</td><td>23.97</td><td>0.12</td><td>2.81</td><td>26.30</td><td>37.06</td><td>0.06</td><td>0.84</td><td>22.42</td><td>31.15</td></tr><tr><td>Co-processor (2025)</td><td>0.88</td><td>12.05</td><td>43.09</td><td>63.12</td><td>0.52</td><td>9.54</td><td>43.16</td><td>58.91</td><td>0.28</td><td>4.41</td><td>32.71</td><td>42.01</td></tr><tr><td>MemGen (2026a)</td><td>0.62</td><td>9.98</td><td>42.16</td><td>41.56</td><td>0.46</td><td>8.70</td><td>42.42</td><td>49.11</td><td>0.29</td><td>4.63</td><td>32.78</td><td>26.85</td></tr><tr><td>ThinkFlow (Ours)</td><td>1.21</td><td>14.84</td><td>45.62</td><td>58.85</td><td>0.92</td><td>13.21</td><td>45.77</td><td>51.81</td><td>0.77</td><td>9.03</td><td>38.89</td><td>39.56</td></tr><tr><td>Long Context (128K)</td><td>0.66</td><td>10.72</td><td>45.89</td><td>21.43</td><td>0.41</td><td>9.20</td><td>43.85</td><td>46.49</td><td>0.43</td><td>7.51</td><td>33.91</td><td>21.13</td></tr><tr><td colspan="11">Explicit Memory Paradigm</td><td></td></tr><tr><td rowspan="15">Qwen3-8B COMEDY (2025) LD-Agent (2025a) THEANINE (2025) LightMem (2026)</td><td>GraphRAG (2024)</td><td>1.15</td><td>13.21</td><td>46.13</td><td>34.17</td><td>0.65</td><td>11.13</td><td>45.17</td><td>42.51</td><td>0.54</td><td>8.37 35.21</td><td></td><td>24.53</td></tr><tr><td>MemTree (2025)</td><td>1.09</td><td>12.97</td><td>45.97</td><td>31.59</td><td>0.59</td><td>10.68</td><td>44.49</td><td>41.76</td><td>0.51</td><td>8.10</td><td>34.97</td><td>23.85</td></tr><tr><td>MemGPT (2023)</td><td>1.06</td><td>12.82</td><td>45.41</td><td>39.68</td><td>0.81</td><td>11.42</td><td>44.68</td><td>46.24</td><td>0.52</td><td>8.23</td><td>34.83</td><td>23.91</td></tr><tr><td>A-Mem (2025b)</td><td>0.94</td><td>13.63</td><td>43.79</td><td>26.85</td><td>0.77</td><td>12.31</td><td>44.62</td><td>44.15</td><td>0.49</td><td>8.07</td><td>34.59</td><td>22.87</td></tr><tr><td>Mem0 (2025)</td><td>0.72</td><td>12.23</td><td>43.12</td><td>45.89</td><td>0.63</td><td>11.57</td><td>43.32</td><td>45.72</td><td>0.51</td><td>8.11</td><td>34.67</td><td>23.41</td></tr><tr><td>MemoryOS (2025)</td><td>0.81</td><td>12.45</td><td>43.51</td><td>46.23</td><td>0.67</td><td>11.73</td><td>43.83</td><td>46.11</td><td>0.53</td><td>8.19</td><td>34.73</td><td>23.77</td></tr><tr><td>MemoChat (2023)</td><td>0.95</td><td>13.73</td><td>42.19</td><td>42.13</td><td>0.77</td><td>12.11</td><td>43.91</td><td>44.53</td><td>0.47</td><td>7.83</td><td>33.87</td><td>22.19</td></tr><tr><td>MemoryBank (2024)</td><td>1.22</td><td>14.91</td><td>43.40</td><td>47.53</td><td>0.88</td><td>13.26</td><td>45.14</td><td>46.33</td><td>0.61</td><td>8.93</td><td>35.53</td><td>25.13</td></tr><tr><td>Rsum (2025b)</td><td>1.17</td><td>14.53</td><td>43.13</td><td>46.87</td><td>0.85</td><td>13.01</td><td>44.87</td><td>45.89</td><td>0.58</td><td>8.71 35.29</td></table>

Table 1: Evaluation of generation performance (%) on conversational datasets. The best result is bolded and the second-best is underlined. \*B-4 = BLEU-4, R-L = ROUGE-L, and Bert = BertScore.

## 4.5 Framework Analysis

The latent memory naturally disentangles static facts from continuously shifting user states, forming interpretable manifolds. Figure 3 visualizes the latent memory space using t-SNE on the PersonaMem bench mark. It reveals a clear structural separation among different cognitive tasks. Notably, representations for "Shared Facts" form a dense, highly localized cluster, reflecting the static nature of objective knowledge. In stark contrast, the manifold for "Track Evolution" spans a wide, continuous region. This visually confirms that our framework successfully tracks shifting user states within a fluid latent space, rather than relying on rigid text contents and summaries.

Expanding the capacity of latent cognitive receptors enables finer-grained feature disentanglement, thereby consistently enhancing personalization. Figure 4 illustrates the impact of scaling the number of latent memory skills (K) on the PersonaMem benchmark with a 1M context. As K increases from 1 to 10, the average accuracy exhibits a robust upward trend. Since the benchmark involves 7 task categories, scaling K to 10 acts as an upper bound to test performance when memory capacity exceeds task complexity. This improvement is particularly pronounced in complex reasoning sub-tasks such as "Revisit Reasons" and "Track Evolution". It demonstrates that allocating more latent skill vectors allows the model to capture and disentangle a richer set of user nuances, effectively boosting the resolution of the memory space without context windows.

<table><tr><td>Length</td><td>Methods</td><td>Revisit Reasons</td><td>Latest Prefs</td><td>Shared Facts</td><td>Track Evolution</td><td>New Ideas</td><td>New Scenarios</td><td>Aligned Recs</td><td>Average</td></tr><tr><td rowspan="7">32K</td><td colspan="9">Closed-source LLMs</td></tr><tr><td>GPT-4o-mini*</td><td>74.00</td><td>18.00</td><td>29.00</td><td>48.00</td><td>16.00</td><td>7.00</td><td>29.00</td><td>31.57</td></tr><tr><td colspan="9">Comparable Methods (Qwen3-8B)</td></tr><tr><td>MemTree</td><td>14.14</td><td>17.06</td><td>20.93</td><td>15.11</td><td>9.68</td><td>14.04</td><td>27.27</td><td>16.89</td></tr><tr><td>A-Mem</td><td>13.03</td><td>11.76</td><td>13.10</td><td>12.16</td><td>12.15</td><td>13.51</td><td>13.64</td><td>12.76</td></tr><tr><td>MemoryBank</td><td>63.64</td><td>23.53</td><td>51.94</td><td>56.12</td><td>11.83</td><td>29.82</td><td>36.36</td><td>39.03</td></tr><tr><td>MemGen</td><td>65.66</td><td>17.65 29.41+5.88</td><td>46.51 55.81+3.87</td><td>64.75 70.50+5.75</td><td>12.90 21.51+8.61</td><td>24.56 31.58+1.76</td><td>32.73 34.55-1.81</td><td>37.82 44.43+5.40</td></tr><tr><td></td><td colspan="9">ThinkFlow-8B (Ours) 67.68+2.02</td></tr><tr><td rowspan="8">128K</td><td>Closed- &amp; Open-source LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o-mini* Claude3.7-Sonnet*</td><td>70.00</td><td>34.00</td><td>55.00</td><td>60.00</td><td>10.00</td><td>33.00</td><td>41.00</td><td>43.29</td></tr><tr><td>Llama3.1-405B*</td><td>57.00 41.00</td><td>9.00 31.00</td><td>25.00 38.00</td><td>45.00 38.00</td><td>28.00 20.00</td><td>29.00 21.00</td><td>20.00 37.00</td><td>30.43 32.29</td></tr><tr><td colspan="9"></td></tr><tr><td colspan="9">Comparable Methods (Qwen3-8B)</td></tr><tr><td>MemTree</td><td>16.73</td><td>22.63</td><td>16.37</td><td>17.60</td><td>18.73</td><td>23.47</td><td>24.93</td><td>20.07</td></tr><tr><td>A-Mem</td><td>12.64</td><td>9.47</td><td>8.19</td><td>9.38</td><td>9.07</td><td>7.98</td><td>11.75</td><td>9.78</td></tr><tr><td>MemoryBank</td><td>53.16 57.25</td><td>38.11</td><td>26.32</td><td>57.77</td><td>15.06</td><td>24.88</td><td>30.95</td><td>35.18</td></tr><tr><td>MemGen ThinkFlow-8B (Ours)</td><td>63.20+5.95</td><td>40.65 50.92+10.27</td><td>22.22 29.24+2.92</td><td>61.58 63.64+2.06</td><td>19.11 20.08+0.97</td><td>26.76 27.23+0.47</td><td>31.81 32.09+0.28</td><td>37.05 40.91+3.86</td></tr><tr><td colspan="9"></td></tr><tr><td rowspan="7">1M</td><td>Closed-source LLMs</td><td>68.00</td><td>39.00</td><td>42.00</td><td>62.00</td><td>12.00</td><td>49.00</td><td>41.00</td><td>44.71</td></tr><tr><td colspan="9">Gemini2.0-Flash*</td></tr><tr><td>Comparable Methods (Qwen3-8B)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MemTree</td><td>11.91</td><td>22.79</td><td>20.83</td><td>11.11</td><td>20.08</td><td>25.08</td><td>22.86</td><td>19.24</td></tr><tr><td>A-Mem</td><td>9.79</td><td>6.64</td><td>8.33</td><td>7.11</td><td>5.23</td><td>5.76</td><td>6.79</td><td>7.09</td></tr><tr><td>MemoryBank</td><td>51.49</td><td>33.46</td><td>34.03</td><td>61.33</td><td>19.12</td><td>29.15</td><td>26.78</td><td>36.48</td></tr><tr><td>MemGen ThinkFlow-8B (Ours) 68.09+15.32</td><td>52.77</td><td>39.32 40.76+1.44</td><td>35.42 38.89+3.47</td><td>63.11 62.22-0.89</td><td>19.53 18.98-1.10</td><td>32.20 33.22+1.02</td><td>29.64 31.43+1.79</td><td>38.86 41.94+2.36</td></tr></table>

Table 2: Accuracy on the PersonaMem benchmark. For comparable methods, the best result is bolded and the second-best is underlined. Baseline results marked with <sup>∗</sup> are retrieved from (Jiang et al., 2025a). Gain (<sup>+·</sup>) and loss (<sup>−·</sup>) represent the relative performance of our ThinkFlow-8B compared to the best comparable method in each task.

<table><tr><td>Methods</td><td>B-4</td><td>R-L</td><td>Bert</td><td>Mauve</td></tr><tr><td>ThinkFlow (Ours)</td><td>2.18</td><td>17.83</td><td>47.77</td><td>77.20</td></tr><tr><td>w/o PLMS</td><td>1.88</td><td>16.54</td><td>45.53</td><td>66.48</td></tr><tr><td>w/o GLC</td><td>1.94</td><td>16.99</td><td>45.59</td><td>64.98</td></tr><tr><td>w/o CAHA</td><td>2.04</td><td>17.45</td><td>44.98</td><td>67.65</td></tr><tr><td>w/o Phase1</td><td>2.00</td><td>17.33</td><td>44.84</td><td>67.87</td></tr><tr><td>w/o Phase2</td><td>2.04</td><td>17.27</td><td>44.87</td><td>69.02</td></tr></table>

Table 3: Ablation study of our method (Qwen3-8B) On CC dataset. More results are show in Appendix D.

Operating exclusively in the latent space eliminates the massive token overhead and computational latency inherent in explicit memory pipelines. Figure 5 illustrates the token-time scaling efficiency across interaction sessions. ThinkFlow demonstrates remarkable efficiency, consuming the least tokens (2,857) and requiring the shortest time (12.1s). In contrast, explicit methods like MemTree and A-Mem suffer from severe computational bottlenecks, generating tens of thousands of hidden tokens and taking nearly 8 times longer to execute. This proves that our latent memory paradigm is not only highly accurate but also exceptionally lightweight and scalable for real-time companionship.

![](images/4577715105c7faf57e47dec5b531a47d46b2d60278855a994a2fab8dbe98e8c6.jpg)  
Figure 3: t-SNE visualization of latent memory across three typical tasks on PersonaMem.

<table><tr><td>Dialogue History &amp; Context</td><td colspan="2">[Relationship: Husband &amp; Wife]</td></tr><tr><td>Session 3 (Past Preference) Session 5 (Relationship Crisis)</td><td colspan="2">User: I saw this amazing jacket in the window display today... I&#x27;m willing to spend the money to look good. User: I can&#x27;t believe you&#x27;ve been seeing someone else behind my back! ... I feel like everything we had has been shattered.</td></tr><tr><td>Current Turn (Preference Shift)</td><td colspan="2">User: &quot;That seems like a lifetime ago. I don&#x27;t feel much like dressing up these days.</td></tr><tr><td colspan="2">Baseline: MemTree (Explicit Retrieval)</td><td>Ours: ThinkFlow (Probabilistic Latent)</td></tr><tr><td colspan="2">MEMORY STATUS Retrieved Nodes (Similarity triggered): • [Node 49]: I saw this amazing jacket.. • [Node 82]: loved trying out new styles...</td><td>MEMORY STATUS Latent State Evolution: Distribution transitions smoothly from [Interest in Fashion] →[Deep Sadness].</td></tr><tr><td colspan="2">AGENT RESPONSE I understand where you’re coming from... maybe we could try something new together, like exploring a new style!</td><td>AGENT RESPONSE I completely understand. When you&#x27;re dealing with so much pain, it&#x27;s completely normal to lose interest in things like dressing up. I’m here for you, take all the time you need.</td></tr><tr><td colspan="2">ANALYSIS × Failure: Trapped by the surface fact “dressing up”. Rigidly retrieves outdated explicit history and entirely misses the user&#x27;s emotional collapse.</td><td>ANALYSIS √Success: Successfully tracks the continuous latent emotion, naturally adapting to the loss of interest without rigid similarity dependency.</td></tr></table>

Table 4: Case study on tracking user evolution on CC dataset.

![](images/e2a230ff9e217d83b705b4c8d775e002bcfc63ec13f6f4d2ca61ee4878990b39.jpg)  
Figure 4: Latent memory scaling on PersonaMem with 1M context. More results are shown in Appendix F.

## 4.6 Case Study

Explicit retrieval often traps models in outdated contexts, whereas latent memory smoothly adapts to profound emotional shifts. Table 4 illustrates a multi-session dialogue where a severe relationship crisis alters the user’s past interests. The explicit baseline MemTree is easily trapped by the surface fact "dressing up", rigidly retrieving outdated history and delivering an emotionally tone-deaf response. In contrast, Think-Flow dynamically tracks the smooth transition of the latent emotional state from initial interest to deep sadness. Consequently, our method generates a highly empathetic and contextually appropriate response without relying on similarity retrieval.

![](images/c596147df7719c0efc4f128e416c0e383b18d33097ab87e8f35dfce5436ea137.jpg)  
Figure 5: Token-time scaling efficiency on CC dataset, where "Hidden" tokens refer to the intermediate memory processed internally by explicit baselines.

## 5 Conclusions

In this work, we propose ThinkFlow, a novel end-toend latent memory framework designed to empower lifelong conversational agents. By departing from traditional explicit textual memory pipelines that suffer from severe information bottlenecks, ThinkFlow dynamically compresses continuous conversational flows into disentangled probabilistic memory skills. Furthermore, our introduced self-supervised test-time evolution paradigm effectively overcomes the static deployment bottleneck, allowing the agent to continuously adapt to evolving user nuances through implicit predictive feedback without requiring manual annotations. Extensive experiments across multiple long-term open-domain conversation datasets and the PersonaMem benchmark demonstrate that ThinkFlow not only significantly outperforms state-of-the-art explicit memory baselines in personalization and response quality, but also achieves remarkable token and temporal efficiency.

## Limitations

While ThinkFlow establishes a highly efficient and selfevolving paradigm for lifelong memory, it opens up several ambitious avenues for future exploration that bound our current scope. First, due to the extraordinary token efficiency and minimal latency of our probabilistic latent memory skills, the current framework processes ultra-long multi-session histories so effortlessly on standard hardware that we have not yet established its upper breakdown limits. Pushing the boundary of this architecture to extreme-scale, life-long contexts—such as infinite multi-modal continuous streams involving lifelong visual and audio inputs—remains an untapped territory, as our continuous latent space is naturally suited for cross-modal fusion rather than being restricted to text.

Second, our self-supervised test-time evolution currently utilizes the "next-user-utterance prediction" as its primary supervisory signal, which already yields robust results. However, given the strong feature disentanglement capabilities observed in our latent manifolds, this mechanism is currently underutilized. Future work should elevate this predictive coding mechanism to anticipate highly complex, long-horizon user behavioral trajectories or to simulate intricate multi-agent societal dynamics. Consequently, our current limitation primarily lies in the fact that the framework’s full cognitive potential has yet to be unleashed across broader, crossmodal, and societal-scale multi-agent scenarios.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China 62576120 and the Major Key Project of PCL2025A11 and PCL2024A08. Thanks for the support provided by OpenI Community (https://openi.pcl.ac.cn).

## References

Ian A Apperly and Stephen A Butterfill. 2009. Do humans have two systems to track beliefs and belieflike states? Psychological review, 116(4):953.

Sanghwan Bae, Donghyun Kwak, Soyoung Kang, Min Young Lee, Sungdong Kim, Yuin Jeong, Hyeri Kim, Sang-Woo Lee, Woomyoung Park, and Nako Sung. 2022. Keep me updated! memory management in long-term conversations. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3769–3787.

Nuo Chen, Hongguang Li, Jianhui Chang, Juhua Huang, Baoyuan Wang, and Jia Li. 2025. Compress to impress: Unleashing the potential of compressive memory in real-world long-term conversations. In Proceedings of the 31st International Conference on Computational Linguistics, pages 755–773.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Andy Clark. 2013. Whatever next? predictive brains, situated agents, and the future of cognitive science. Behavioral and brain sciences, 36(3):181–204.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. 2026. Lightmem: Lightweight and efficient memory-augmented generation. In The Fourteenth International Conference on Learning Representations.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, et al. 2025. Memory in the age of ai agents. arXiv preprint arXiv:2512.13564.

Jihyoung Jang, Minseong Boo, and Hyounghun Kim. 2023. Conversation chronicles: Towards diverse temporal and relational dynamics in multi-session conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13584–13606.

Jihyoung Jang, Taeyoung Kim, and Hyounghun Kim. 2024. Mixed-session conversation with egocentric memory. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 11786– 11815.

Bowen Jiang, Zhuoqun Hao, Young Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo Jose Taylor, and Dan Roth. 2025a. Know me, respond to me: Benchmarking LLMs for dynamic user profiling and personalized responses at scale. In Second Conference on Language Modeling.

Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, et al. 2025b. Personamem-v2: Towards personalized intelligence via learning implicit user personas and agentic memory. arXiv preprint arXiv:2512.06688.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory OS of AI agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25972–25981. Association for Computational Linguistics.

Cai Ke, Yiming Du, Bin Liang, Yifan Xiang, Lin Gui, Zhongyang Li, Baojun Wang, Yue Yu, Hui Wang, Kam-Fai Wong, et al. 2025. Flexibly utilize memory for long-term conversation via a fragment-thencompose framework. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 21130–21147.

Cai Ke, Bin Liang, Xin Liu, Yue Yu, Hui Wang, and Ruifeng Xu. 2026a. Dynamic memory forest: Constructing and tracing conversational trajectories for long-term conversation. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 767–777.

Cai Ke, Jiangyue Yan, Han Zhang, Liu Xin, Zike Yuan, Yue Yu, Hui Wang, and Ruifeng Xu. 2026b. Interactive memory learning for long-term conversations. In Findings of the Association for Computational Linguistics: EMNLP 2026.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. 2026. LLMs get lost in multi-turn conversation. In The Fourteenth International Conference on Learning Representations.

Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang Wang, and Tat-Seng Chua. 2025a. Hello again! LLMpowered personalized agent for long-term dialogue. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5259–5276.

Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, et al. 2025b. Seek in the dark: Reasoning via test-time instancelevel policy gradient in latent space. arXiv preprint arXiv:2505.13308.

Bin Liang, Cai Ke, Runcong Zhao, Qinglin Zhu, Lin Gui, Yue Yu, Hui Wang, Ruifeng Xu, and Kam-Fai Wong. 2026. Meta-memory for large language models. IEEE Transactions on Audio, Speech and Language Processing.

CY Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text Summarization Branches Out: Proceedings of the ACL-04 Workshop, Barcelona, Spain, pages 74–81.

Luyang Liu, Jonas Pfeiffer, Jiaxing Wu, Jun Xie, and Arthur Szlam. 2025. Deliberation in latent space via differentiable cache augmentation. In Forty-second International Conference on Machine Learning.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024a. Lost in the middle: How language models use long contexts. Transactions ofthe Associationfor Computational Linguistics, 12:157–173.

Zhiwei Liu, Weiran Yao, Jianguo Zhang, Liangwei Yang, Zuxin Liu, Juntao Tan, Prafulla K Choubey, Tian Lan, Jason Wu, Huan Wang, et al. 2024b. Agentlite: A lightweight library for building and advancing task-oriented llm agent system. arXiv preprint arXiv:2402.15538.

Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yulan He, Di Yin, Xing Sun, and Yunsheng Wu. 2023. Memochat: Tuning llms to use memos for consistent long-range open-domain conversation. arXiv preprint arXiv:2308.08239.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870.

Kai Mei, Xi Zhu, Wujiang Xu, Wenyue Hua, Mingyu Jin, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, and Yongfeng Zhang. 2024. Aios: Llm agent operating system. arXiv preprint arXiv:2403.16971.

Beren Millidge, Anil Seth, and Christopher L Buckley. 2021. Predictive coding: a theoretical and experimental review. arXiv preprint arXiv:2107.12979.

Kai Tzu-iunn Ong, Namyoung Kim, Minju Gwak, Hyungjoo Chae, Taeyoon Kwon, Yohan Jo, Seungwon Hwang, Dongha Lee, and Jinyoung Yeo. 2025. Towards lifelong dialogue agents via timeline-based memory management. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8631–8661.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. 2023. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. 2021. Mauve: Measuring the gap between neural text and human text using divergence frontiers. Advances in Neural Information Processing Systems, 34:4816–4828.

Alireza Rezazadeh, Zichao Li, Wei Wei, and Yujia Bao. 2025. From isolated conversations to hierarchical schemas: Dynamic tree memory representation for LLMs. In The Thirteenth International Conference on Learning Representations.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. 2024. Raptor: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations.

Dana Schneider, Rebecca Lam, Andrew P Bayliss, and Paul E Dux. 2012. Cognitive load disrupts implicit theory-of-mind processing. Psychological science, 23(8):842–847.

Juntao Tan, Liangwei Yang, Zuxin Liu, Zhiwei Liu, Rithesh RN, Tulika Manoj Awalgaonkar, Jianguo Zhang, Weiran Yao, Ming Zhu, Shirley Kokane, et al. 2025. Personabench: Evaluating ai models on understanding personal information through accessing (synthetic) private user data. In Findings of the Association for Computational Linguistics: ACL 2025, pages 878–893.

Piaohong Wang, Motong Tian, Jiaxian Li, Yuan Liang, Yuqing Wang, Qianben Chen, Tiannan Wang, Zhicong Lu, Jiawei Ma, Yuchen Eleanor Jiang, et al. 2025a. O-mem: Omni memory system for personalized, long horizon, self-evolving agents. arXiv eprints, pages arXiv–2511.

Qingyue Wang, Yanhe Fu, Yanan Cao, Shuai Wang, Zhiliang Tian, and Liang Ding. 2025b. Recursively summarizing enables long-term dialogue memory in large language models. Neurocomputing, page 130193.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations.

Derong Xu, Yi Wen, Pengyue Jia, Yingyi Zhang, Yichao Wang, Huifeng Guo, Ruiming Tang, Xiangyu Zhao, Enhong Chen, Tong Xu, et al. 2025a. Towards multi-granularity memory association and selection for long-term conversational agents. arXiv preprint arXiv:2505.19549.

Jing Xu, Arthur Szlam, and Jason Weston. 2022. Beyond goldfish memory: Long-term open-domain conversation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5180–5197.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025b. A-mem: Agentic memory for llm agents. In Advances in Neural Information Processing Systems.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. 2025c. Softcot: Soft chain-of-thought for efficient reasoning with llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational

Linguistics (Volume 1: Long Papers), pages 23336– 23351.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Xinlei Yu, Zhangquan Chen, Yongbo He, Tianyu Fu, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, Zhe Cao, Jie Xu, et al. 2026. The latent space: Foundation, evolution, mechanism, ability, and outlook. arXiv preprint arXiv:2604.02029.

Guibin Zhang, Muxin Fu, and Shuicheng YAN. 2026a. Memgen: Weaving generative latent memory for selfevolving agents. In The Fourteenth International Conference on Learning Representations.

Qiang Zhang, Jason Naradowsky, and Yusuke Miyao. 2023. Mind the gap between conversations for improved long-term dialogue generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10735–10762.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Yingyi Zhang, Junyi Li, Wenlin Zhang, Pengyue Jia, Xianneng Li, Yichao Wang, Derong Xu, Yi Wen, Huifeng Guo, Yong Liu, and Xiangyu Zhao. 2026b. Evoking user memory: Personalizing LLM via recollection-familiarity adaptive retrieval. In The Fourteenth International Conference on Learning Representations.

Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2025. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47.

Zeyu Zhang, Rui Li, Xiaoyan Zhao, Yang Zhang, Wenjie Wang, Xu Chen, and Tat-Seng Chua. 2026c. Nextmem: Towards latent factual memory for llmbased agents. arXiv preprint arXiv:2603.15634.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731.

## A Dataset Information

Long-Term Open-Domain Conversation. To comprehensively assess the model’s proficiency in turnby-turn response generation during prolonged interactions, we utilize three representative long-term opendomain conversation datasets: Conversation Chronicles (CC) (Jang et al., 2023), Multi-Session Chat (MSC) (Xu et al., 2022), and GapChat (GC) (Zhang et al., 2023): The primary motivation for selecting these specific datasets lies in their human-centric construction methodology. They are meticulously curated through extensive crowdsourcing involving real human participants, ensuring that the dialogue flows, topic transitions, and relationship developments accurately mirror the complexity of real-world open-domain social dynamics. Consequently, evaluating on these datasets is essential for verifying whether Large Language Models (LLMs) can generate natural, coherent, and highly engaging responses that genuinely align with human expectations in lifelong conversational scenarios. Specifically, CC focuses on temporal dynamics and intricate relationship evolutions over extensive sessions. MSC provides a massive corpus of authentic multi-session chats where speakers incrementally learn and recall each other’s historical interests. GC introduces the critical dimension of realistic time gaps (ranging from minutes to years) between sessions, challenging the agent to emulate a human-like perception of the passage of time.

<table><tr><td>Datasets</td><td># of Sessions</td><td># of Episodes</td><td># of Turns</td><td>Avg. Turns per Session</td><td>Avg. Turns per Episode</td></tr><tr><td>CC</td><td>1M</td><td>200K</td><td>11.7M</td><td>11.70</td><td>58.50</td></tr><tr><td>MSC</td><td>16K</td><td>5K</td><td>214K</td><td>13.38</td><td>42.80</td></tr><tr><td>GC</td><td>2.65K</td><td>0.65K</td><td>28.13K</td><td>10.62</td><td>43.28</td></tr></table>

Table 5: The statistics of CC, MSC and GC datasets.
<table><tr><td>Characteristic</td><td>32K Corpus</td><td>128K Corpus</td><td>1M Corpus</td></tr><tr><td colspan="4">Basic Statistics</td></tr><tr><td>Total Evaluation Samples</td><td>589</td><td>2,727</td><td>2,674</td></tr><tr><td>Avg. Query Length (Tokens)</td><td>464.6</td><td>416.3</td><td>415.1</td></tr><tr><td>Avg. Memory Length (Tokens)</td><td>24,193.2</td><td>151,851.0</td><td>911,733.4</td></tr><tr><td colspan="4">Distribution by Evaluated Skill (Query Type)</td></tr><tr><td>Recall Shared Facts</td><td>129</td><td>171</td><td>144</td></tr><tr><td>Acknowledge Latest Prefs</td><td>17</td><td>866</td><td>768</td></tr><tr><td>Track Evolution</td><td>139</td><td>341</td><td>225</td></tr><tr><td>Revisit Reasons</td><td>99</td><td>269</td><td>235</td></tr><tr><td>Aligned Recommendations</td><td>55</td><td>349</td><td>280</td></tr><tr><td>Suggest New Ideas</td><td>93</td><td>518</td><td>727</td></tr><tr><td>New Scenarios</td><td>57</td><td>213</td><td>295</td></tr></table>

Table 6: The statistics of PersonaMem benchmark across different memory corpus scales.

Personalized Memory Question Answering. To further evaluate the crucial role of personalized memory—specifically, the model’s capability to accurately track and reason over dynamic user states across varying context lengths—we use the PersonaMem benchmark (Jiang et al., 2025a). While open-domain generation tests general conversational fluency, maintaining true lifelong companionship fundamentally relies on this personalized memory tracking. This benchmark includes about 15 real-world interaction scenarios (such as medical advice, travel planning, and various recommendations), simulating dynamic, long-term conversations between users and AI assistants. As shown in Table 6, we divide the benchmark into three scales—32K, 128K, and 1M—to test the model’s retrieval and reasoning limits. As the memory size grows, the average dialogue history expands dramatically from about 24,000 tokens to over 910,000 tokens, while the user’s query length stays short at around 400 tokens. This short query, ultralong memory” setup strongly challenges the model’s ability to find precise information amidst noise. Furthermore, the benchmark tests 7 specific skills. As the table shows, these tasks range from simple fact retrieval (e.g., Recall Shared Facts) to complex reasoning tasks that require tracking changing preferences (Track Evolution) or generalizing to entirely new situations (Suggest New Ideas/New Scenarios). This diverse mix of tasks ensures we evaluate the model’s true understanding” of a user profile, rather than memorizing long texts.

## B Compared Baselines

To systematically evaluate the effectiveness of Think-Flow, we compare it against a diverse set of state-of-theart baselines. Following the taxonomy established in our Related Work, these baselines are categorized into the Explicit Memory Paradigm and the Latent Memory Paradigm.

## B.1 Explicit Memory Paradigm

This paradigm predominantly relies on discrete, humanreadable text to construct and manage historical interactions. Based on their core mechanisms, we further divide them into three sub-categories:

## B.1.1 Structured Retrieval-Augmented Generation

These methods focus on organizing past conversational logs into structured topologies, such as graphs or trees, to facilitate complex context retrieval.

• GraphRAG (Edge et al., 2024): GraphRAG builds an entity knowledge graph from the source text and groups related entities into hierarchical community summaries. When answering a query, it retrieves information from these structured summaries, making it particularly effective at answering global questions that require understanding the entire document corpus.

• MemTree (Rezazadeh et al., 2025): MemTree organizes conversational history into a dynamic, treestructured hierarchy. Each node within the tree stores aggregated text and its semantic embedding at varying levels of abstraction. It continuously updates this structure by comparing new information with existing nodes, enabling the agent to handle complex reasoning tasks over extended contexts.

## B.1.2 Agentic Memory Management

Approaches in this category utilize autonomous agent architectures or multi-tiered storage systems to actively read, write, and govern memory blocks.

• MemGPT (Packer et al., 2023): Inspired by traditional operating systems, MemGPT introduces a virtual context management system. It creates the illusion of an infinite context window by intelligently paging relevant historical data between a large external storage disk and the LLM’s limited working memory.

• Mem0 (Chhikara et al., 2025): Mem0 operates through a two-step pipeline consisting of extraction and updating. It first identifies essential facts from conversational turns and subsequently employs tool-calling mechanisms to autonomously determine whether to insert, modify, remove, or entirely disregard the incoming information.

• A-Mem (Xu et al., 2025b): Drawing inspiration from the Zettelkasten note-taking method, A-Mem converts conversational interactions into atomic memory notes enriched with specific tags and keywords. It then utilizes embedding-based similarity searches to precisely retrieve relevant historical contexts.

• MemoryOS (Kang et al., 2025): Emulating a computer operating system, MemoryOS organizes interaction data across short-term, mid-term, and long-term storage tiers. It efficiently manages memory by clustering topic-related dialogues into structured segments and individual pages.

## B.1.3 Memory-Augmented Generation

These frameworks are designed to dynamically extract discrete factual points or generate condensed textual summaries to build plug-and-play historical memory banks.

• MemoChat (Lu et al., 2023): MemoChat utilizes instruction-tuning to train language models to generate and consult internal "memos." It relies on a structured cycle of memorization, retrieval, and response generation to maintain logical consistency during extended conversations.

• MemoryBank (Zhong et al., 2024): Inspired by human cognitive processes, MemoryBank continuously archives conversation logs and distills them into hierarchical event summaries. This progressive summarization enables the agent to gradually align with the user’s specific personality.

• Rsum (Wang et al., 2025b): Rsum introduces a recursive summarization framework. It sequentially merges the previously stored memory state with the latest dialogue segments to produce an updated summary, thereby preserving temporal continuity.

• COMEDY (Chen et al., 2025): Moving away from conventional external retrieval modules, COMEDY employs a unified model to concurrently handle memory generation, compression, and response formulation. It condenses user dynamics and historical events into a highly compact textual representation.

• LD-Agent (Li et al., 2025a): LD-Agent addresses long-term personalization through a decoupled, modular architecture. It distinctly isolates the independent processes of perceiving events, extracting user personas, and generating the final responses.

• THEANINE (Ong et al., 2025): In contrast to systems that routinely overwrite or discard obsolete data, THEANINE deliberately preserves all historical records. This inclusive strategy ensures that shifting user habits and outdated—yet contextually valuable—behavioral patterns are not permanently lost.

• LightMem (Fang et al., 2026): Inspired by the Atkinson-Shiffrin cognitive model, LightMem organizes interactions into a three-stage memory architecture: sensory, short-term, and long-term memory. It rapidly filters irrelevant information and consolidates interactions into topic-based groups, significantly reducing computational overhead while maintaining effective historical retrieval.

## B.2 Latent Memory Paradigm

Unlike explicit methods constrained by the textual bottleneck, these approaches encode multi-session experiences directly into continuous vector spaces to support implicit agentic reasoning.

• SoftCoT (Xu et al., 2025c): SoftCoT utilizes a lightweight assistant model to generate instancespecific "soft thought tokens" in a continuous space.

These latent representations are then mapped into the primary language model’s representation space via a trainable projection module, enabling intermediate reasoning without altering the main model’s parameters.

• Co-processor (Liu et al., 2025): Co-processor augments a frozen language model with an offline, asynchronous module that operates directly on the model’s key-value (KV) cache. It distills additional computational steps into latent embeddings within the KV-cache, improving the fidelity of subsequent decoding without generating discrete text tokens.

• MemGen (Zhang et al., 2026a): MemGen introduces a dynamic generative memory framework featuring a memory trigger and a memory weaver. It monitors the agent’s current state to dynamically construct latent token sequences, functioning as machine-native memory that seamlessly interweaves continuous memory representations with the reasoning process.

## C Implementation Details

As showin Table 7, LoRA targets the query, key, value, and output projection layers (q\_proj, k\_proj, v\_proj, o\_proj) with rank $r \ = \ 8$ and $\alpha = 1 6$ Within the memory components, the cross-attention module in the Probabilistic Latent Memory Skills (PLMS) utilizes 8 attention heads. In the Gated Latent Consolidator (GLC), the intermediate projection size of the MLP gate is set to half of the model’s hidden dimension $( D / 2 )$ . For the Context-Aware Hyper-Aligner (CAHA), the bottleneck rank is set to $r = 8$

During training, we optimize the framework using the AdamW optimizer with a learning rate of $2 \times 1 0 ^ { - 5 }$ and apply gradient clipping with a maximum norm of 1.0. For the loss constraints, the KL divergence regularization weight α is set to 0.1, and the minimum entropy confidence prior weight β for Phase 2 is also set to 0.1. During inference, responses are generated using nucleus sampling with top-p = 0.8, a temperature of $\tau = 0 . 3$ and a maximum sequence length of 1024 tokens.

## D Ablation Study

Due to space constraints in the main text, we present the comprehensive ablation study results on the MSC and GC datasets in this section. Table 8 details the performance of our proposed ThinkFlow and its variants on these two additional datasets. Consistent with the observations on the CC dataset, removing any key architectural component (PLMS, GLC, or CAHA) or training phase (Phase1 or Phase2) leads to a performance drop across most metrics. This further demonstrates the robustness of our design choices and the generalizability of our model’s capabilities across different evaluation domains.

Algorithm 1: ThinkFlow: Self-Supervised Test-  
Time Evolution   
Input: Long-term dialogue stream   
$\mathcal { D } = \{ \boldsymbol { S } _ { 1 } , \boldsymbol { S } _ { 2 } , \ldots , \boldsymbol { \bar { \mathbf { \theta } } } , \boldsymbol { S } _ { N } \}$ , Teacher LLM M<sub>T</sub>,   
Student LLM M<sub>S</sub> (with LoRA), Memory   
modules (PLMS, GLC, CAHA)   
Output: Optimized Student LLM M and Memory   
modules   
1 Initialize latent memory skills $S _ { 0 }  \emptyset ;$   
/\* Phase 1: Teacher-Guided Latent Alignment   
\*/   
2 for epoch $e = 1$ to E do   
3 $\dot { S } _ { t - 1 }  S _ { 0 } ;$   
4 for each turn $( x _ { t } , a _ { t } )$ in cold-start session $\scriptstyle { S _ { 1 } }$ do   
5 $P _ { T }  \dot { \mathcal { M } _ { T } } ( y \mid ^ { ^ { \prime } } H _ { < t } , x _ { t } ) ; / /$ Target dist.   
from explicit history   
6 $S _ { a l i g n e d }  \mathrm { C A H A } ( S _ { t - 1 } , x _ { t } ) ; / /$ Dynamic   
latent translation   
7 $P _ { S } , E _ { t }  { \mathcal { M } } _ { S } ( y \mid S _ { a l i g n e d } , x _ { t } )$ ;   
// Student prediction   
8 $\hat { S } _ { n e w } , \mathcal { L } _ { K L } \gets \mathrm { P L M S } ( E _ { t } ) ; \quad / /$ Compress   
context   
9 $S _ { t } \gets \mathrm { G L C } ( \hat { S } _ { n e w } , S _ { t - 1 } ) ; / /$ Consolidate   
memory   
10 $\mathcal { L } _ { P h a s e 1 }  \mathcal { D } _ { K L } ( P _ { T } \parallel P _ { S } ) + \alpha \mathcal { L } _ { K L } ;$   
11 $\nabla \mathcal { L } _ { P h a s e 1 }  \mathbf { U p d a t e } \mathcal { M } _ { S } , \mathbf { P L M S } , \mathbf { G L C } ,$   
CAHA;   
12 end   
13 end   
/\* Phase 2: Next-User-Utterance Prediction   
(Online Evolution) \*/   
14 for each session ${ { S } _ { i } } \in \{ { { S } _ { 2 } } , \ldots , { { S } _ { N } } \}$ do   
15 for each turn $( x _ { t } , a _ { t } )$ and next user query $x _ { t + 1 }$ in   
$\boldsymbol { S } _ { i }$ do   
16 $S _ { a l i g n e d }  \mathbf { C A H A } ( S _ { t - 1 } , x _ { t } ) ;$   
17 $E _ { t } \gets \mathcal { M } _ { S } ( S _ { a l i g n e d } , x _ { t } , a _ { t } ) ;$   
18 $\hat { S } _ { n e w } , \mathcal { L } _ { K L } \gets \mathrm { P L M S } ( E _ { t } ) ;$   
19 $S _ { t } \gets \mathrm { G L C } ( \hat { S } _ { n e w } , S _ { t - 1 } ) ;$   
20 $P _ { p r e d }  \dot { \mathcal { M } } _ { S } ( x _ { t + 1 } \mid S _ { t } , x _ { t } , a _ { t } )$ ;   
// Implicit prediction feedback   
21 $\mathcal { L } _ { C E } \gets \mathrm { C r o s s E n t r o p y } ( P _ { p r e d } , x _ { t + 1 } ) ;$   
22 $\mathcal { L } _ { C o n f }  - \sum P _ { p r e d } \tilde { \log } P _ { p r e d } ;$   
// Min-Entropy regularization   
23 $\mathcal { L } _ { P h a s e 2 }  \mathcal { L } _ { C E } + \beta \mathcal { L } _ { C o n f } + \alpha \mathcal { L } _ { K L } ;$   
24 $\nabla \mathcal { L } _ { P h a s e 2 } $ Update M , PLMS, GLC,   
CAHA;   
25 end   
26 end

## E Prompts

Unlike traditional explicit memory pipelines that suffer from the text bottleneck by appending retrieved text summaries, ThinkFlow maintains a minimal textual context (Figure 6). Instead, historical user states are dynamically injected as continuous probabilistic latent memory skills directly into the embedding space.

## F Parameter Analysis

Scaling latent capacity is increasingly vital for ultralong interactions to prevent feature blending. Figure 7 analyzes the scaling effect of latent memory skills (K) across 32K, 128K, and 1M context lengths. While shorter contexts (32K) maintain relatively high accuracy even with fewer skills, the ultra-long 1M context scenario exhibits a sharp and robust upward trend as K increases from 1 to 10. This reveals a critical insight: as the dialogue history grows exponentially, allocating a larger set of disentangled latent vectors becomes essential to independently capture expanding user nuances without semantic interference.

<table><tr><td>Category</td><td>Hyper-parameter</td><td>Value</td></tr><tr><td>LoRA</td><td>Target Modules</td><td>q_proj, k_proj, v_proj, o_proj</td></tr><tr><td></td><td>Rank (r)</td><td>8</td></tr><tr><td></td><td>Alpha (α)</td><td>16</td></tr><tr><td>Memory Architecture</td><td>PLMS Attention Heads</td><td>8</td></tr><tr><td></td><td>GLC MLP Intermediate Size</td><td> $D / 2$ </td></tr><tr><td></td><td>CAHA Bottleneck Rank</td><td>8</td></tr><tr><td></td><td>Number of Skills (K)</td><td>10</td></tr><tr><td>Training</td><td>Optimizer</td><td>AdamW</td></tr><tr><td></td><td>Learning Rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td></td><td>Max Gradient Norm</td><td>1.0</td></tr><tr><td></td><td>KL Divergence Weight (α)</td><td>0.1</td></tr><tr><td></td><td>Min Entropy Weight (β)</td><td>0.1</td></tr><tr><td></td><td>Cold-Start Epochs</td><td>3</td></tr><tr><td>Inference</td><td>Decoding Strategy</td><td>Nucleus Sampling</td></tr><tr><td></td><td>Top-p</td><td>0.8</td></tr><tr><td></td><td>Temperature (τ)</td><td>0.3</td></tr><tr><td></td><td>Max Sequence Length</td><td>1024</td></tr></table>

Table 7: Detailed hyper-parameter configurations for ThinkFlow.
<table><tr><td rowspan="2">Methods</td><td colspan="4">MSC</td><td colspan="4">GC</td></tr><tr><td>B-4</td><td>R-L</td><td>Bert</td><td>Mauve</td><td>B-4</td><td>R-L</td><td>Bert</td><td>Mauve</td></tr><tr><td>ThinkFlow (Ours)</td><td>1.12 </td><td>14.33</td><td>46.72</td><td>66.48</td><td>1.10</td><td>14.21</td><td>44.82</td><td>65.26</td></tr><tr><td>w/o PLMS</td><td>0.96</td><td>13.33</td><td>45.06</td><td>65.09</td><td>0.93</td><td>9.32</td><td>37.88</td><td>45.19</td></tr><tr><td>w/o GLC</td><td>1.03</td><td>13.70</td><td>45.17</td><td>65.83</td><td>1.03</td><td>13.70</td><td>45.17</td><td>65.83</td></tr><tr><td>w/o CAHA</td><td>1.10</td><td>14.21</td><td>44.82</td><td>65.26</td><td>1.04</td><td>10.25</td><td>39.89</td><td>52.23</td></tr><tr><td>w/o Phase1</td><td>1.07</td><td>13.98</td><td>44.71</td><td>64.22</td><td>1.07</td><td>13.98</td><td>44.71</td><td>64.22</td></tr><tr><td>w/o Phase2</td><td>1.09</td><td>14.03</td><td>44.86</td><td>64.35</td><td>1.09</td><td>14.03</td><td>44.86</td><td>64.35</td></tr></table>

Table 8: Ablation study of our method (Qwen3-8B) on MSC and GC datasets, where w/o means without.

![](images/6d6edced574964d87548a838ce6ca849814003577b5adcb591882664ca9f6e53.jpg)  
Figure 6: Prompt structure for generating personalized agent responses in ThinkFlow.

(a) Scaling Trend of Latent Memory Skills  
![](images/059b8718e7f0464f06e28ec2b3e3ebc30acc8b478582dbfc8b2fe43609f5e404.jpg)  
(b) Detailed Performance at Different Tasks

![](images/d05102ca6fafc22b4943528cda68c1d6cb36e4221ed0f0331c649d32e0c35e9c.jpg)  
Figure 7: Latent memory scaling on PersonaMem.

## G Algorithm Walkthrough

Algorithm 1 outlines the two-phase training paradigm of ThinkFlow. Phase 1 (Lines 2-12) addresses the coldstart problem by distilling knowledge from a text-based Teacher LLM. The teacher accesses the full explicit context to generate a target distribution. In contrast, the Student LLM relies solely on the dynamically translated latent memory skills (via CAHA). The student updates its latent state through the PLMS and GLC modules and aligns with the teacher by minimizing the KL divergence. Phase 2 (Lines 13-25) enables continuous, label-free evolution. Operating purely in the latent space, the agent uses implicit predictive feedback to anticipate the user’s next utterance $\left( x _ { t + 1 } \right)$ based on the consolidated memory. The framework continuously optimizes itself through a cross-entropy loss against the actual user reaction, regularized by a minimum entropy confidence prior.