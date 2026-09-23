# Combining Hierarchical Cognitive Process with Process Supervision for Interpretable Scene Safety Understanding

Zhiyun Jiang, Hanyong Wang, Binbin Liang, Yu Xie, Zhengjie Wang, Menglong Yang, Wei Li

A R T I C L E I N F O

Keywords:   
scene understanding   
cognitive process   
large language model   
process label

## A BS T RA C T

Scene safety understanding plays a life-or-death role in situational awareness in various critical domains. Traditional methods that rely on learning direct mappings between scenes and safety levels often lack interpretability, limiting their reliability in critical applications. An efective approach to overcoming this challenge lies in interpreting human cognitive processes and equipping machine models with analogous cognitive capabilities. This work explores an efective way of integrating scene safety cognitive process modeling and process supervision. Specifically, we first construct a hierarchical cognitive safety structure, which motivates the development of a novel, high-quality scene safety understanding dataset based on multi-step reasoning with process labels. This dataset serves both as a benchmark and a resource to improve the safety reasoning capabilities of Large Language Models (LLMs), while also enabling a granular analysis of intermediate reasoning steps through information flow and saliency-based techniques. Building upon this foundation, we introduce a modular and flexible process supervision framework that reflects the hierarchical nature of human cognition. This framework leverages LLMs as the core architecture and incorporates Low-Rank Adaptation(LoRA) and Mixture-of-Experts (MoE) strategies to enable specialization and collaboration among expert modules, each tasked with specific sub-processes of the overall reasoning chain. Systematic experimental evaluations and analyses confirm that our framework exhibits superior interpretability and performance characteristics compared to traditional approaches.

## 1. Introduction

Safety constitutes a fundamental cognitive dimension in scene understanding frameworks, serving as a critical indicator of situational awareness in dynamic environmentsShree, Asfora, Zheng, Hong, Banfi and Campbell (2021). The significance of scene safety understanding is paramount across diverse applications, including safety assessments on construction sitesChen, Hou, Wu, Zhang, Zou, Moon and Bhuiyan (2024) and public transportation scenariosLi, Fang and Xue (2024). The rapid advancement in machine intelligence technologies has led to a growing interest in the automatic understanding of scene safety. The primary objective of automatic scene safety understanding is to leverage machine learning techniques to extract features of interest from a given scene, analyze the relationships and behaviors of these features using computer vision and natural language processing methods, and ultimately infer the safety status of the scene.

Current machine learning models predominantly employ an outcome-supervised training paradigm, wherein the models are trained to reconstruct a holistic scene safety assessmentWang, Malawade, Zhou, Yu and Al Faruque (2024b). However, scenes invariably encompass a variety of entities whose appearances and interactions exhibit significant stochasticity. The intricate relationships among these scene components extend beyond simple compositionalityAn, Zhu, Panaitescu-Liess, Mummadi and Huang (2024). For machine learning models, the inherent complexity and variability of real-world scenes pose substantial challenges in information extraction and integration, relational reasoning, and logical induction and deduction. Furthermore, the heavy reliance of machine learning technologies on statistical learning as their core frameworkKhaleel, Ahmed and Alsharif (2023) frequently leads to issues such as latency and lack of interpretability in cognition, assessment, and decision-making, thereby complicating the assurance of the reliability of machine intelligence.

To achieve accurate scene safety understanding, it is essential that machine intelligence incorporates human cognitive capabilities. Humans possess extensive visual experience, common sense knowledge, and semantic understandingZhu, Gao, Fan, Huang, Edmonds, Liu, Gao, Zhang, Qi, Wu et al. (2020), which enable them to perform in-depth analysis and relational reasoning about complex scenes. Reliable and efective human-machine collaborative cognition requires the seamless integration of human cognitive information into machine intelligence systems, coupled with the provision of timely feedback from the systemDeng, Ji, Rainey, Zhang and Lu (2020). However, efectively articulating human cognitive processes and designing interaction and reasoning mechanisms that are both observable and controllable within the context of human-machine collaborative scene safety understanding remain critical challenges. The global workspace theory suggests that the brain solves complex problems through the collaboration of multiple submodules that perform specific tasksBaars (2006). If machine learning systems can be decomposed into modular subcomponents that map to diferent analytical levels within the human cognitive hierarchy, this facilitates targeted injection of human knowledge into relevant stages of the model.

In this study, we construct a step-wise modular machine cognition model, with each component focusing on diferent aspects of scene safety understanding analogous to stages in the human cognitive process. We introduce a cognitive hierarchy model that mimics the human cognitive process as a supervisory signal to perform process supervision on the machine cognition model. Unlike conventional outcome supervision approaches, process supervision models receive feedback from each stage of the cognitive hierarchy, allowing for targeted debugging and adjustment of relevant modules in the machine cognition model. Specifically, the machine cognition model can begin with the impact of lowerlevel object/instance cognition on scene safety understanding from the cognitive hierarchy, progressively acquiring relationship mining capabilities from the cognitive hierarchy, and ultimately achieving cognitive hierarchy-guided overall scene safety understanding. In other words, each step or stage of the machine cognition model can receive feedback from the cognitive hierarchy. Thus, under the guidance of the cognitive hierarchy model, scene safety understanding by the machine cognition model is accomplished progressively. The step-wise injection of human knowledge through the cognitive hierarchy enables interpretable debugging capabilities at each stage of the machine cognition model and promotes model evolution in accordance with human cognitive processes.

The motivation for this study lies in the design of scene safety understanding capabilities by emulating the hierarchical and sequential nature of human scene perception and safety assessment. This approach seeks to develop scene safety interpretation abilities via process-supervised learning, inspired by and modeled after human cognitive mechanisms. The specific contributions of this research can be summarized as follows:

1. We propose a hierarchical cognitive structure for scene safety and further create a scene safety understanding dataset based on multi-step reasoning with process labels (SSUPL), which reflects this cognitive mechanism.

2. We demonstrate that SSUPL-based scene safety understanding in LLMs exhibits human-like hierarchical cognitive mechanisms from an information flow perspective, and further analyze the key factors afecting the final safety prediction in LLMs.

3. We propose a process-supervised training framework that reflects the cognitive hierarchy by combining Mixture-of-Experts (MoE) and Low-Rank Adaptation (LoRA). We conduct a large number of experiments based on this framework on diferent LLMs, and the results show that our approach outperforms the traditional paradigm.

## 2. Related Works

## 2.1. Scene Safety Understanding

Traditional machine learning-based approaches to scene safety typically begin by having an expert define a collection of categories for some common pitfalls, hazards, or anomalies in a vertical domain. Then, these predefined categories in a specific scene are identified or detected using discriminative models of computer vision to determine whether the scene is safe or not based on their presence or absenceLoo, Fan, Lian and Zhang (2023). In addition, some studies consider the relationships between entities based on entities perception and introduce scene graphs to enhance the model reasoning capability, thus improving the model performanceZhang, Wang, Wang, Sun and Zhao (2022). Benchmarks for road trafic safetyKung, Yang, Pao, Lu, Chen, Lu and Chen (2023) and construction site safetyXuehui, Li, Zuguang, Chengzhi, Pengfei and Zhiwei (2021) have emerged. Overall, these approaches focus on improving the model’s recognition or detection performance, and are unable to understand the safety of the scene from a macroscopic perspective. The model in this case learns only the mapping relationship between objects and categories in the scene or between the scene and the safety judgment, and the interpretability of the final results obtained is poor.

Recently, under the development wave of generative artificial intelligence, LLMs, multi-modal large models(MLLMs), etc., have started to contribute to the development of scene safety understanding. Thanks to their extensive knowledge base, they are able to understand input scenes and perform causal reasoning for safetyZheng, Abdel-Aty, Wang, Wang and Ding (2023). Currently, the vast majority of research addresses the field of road trafic safetyLohner, Compagno, Francis and Oltramari (2024). Overall, most of the studies are currently at the stage of proposing generative model benchmarks for vertical domains. Their methods of constructing datasets require specialised vertical domain knowledge and are not universal. They do provide safety reasoning processes in textual form to interpret the results, however, they lack the perspective of interpreting from within the model.

## 2.2. Process Supervision

Process supervision refers to continuous monitoring, guidance and adjustment by providing feedback on the various stages of training a model to accomplish a particular task. Broadly speaking, autoregressive models for language generation tasks embody the idea of process supervision during the training process. However, this is mainly attributed to the directional and process properties of the language and the next token prediction of the autoregressive model. Narrow process supervision encompasses three key points: task decomposition, intermediate oversight, and transparency and error correction, the last of which is not present in the broad process supervision mentioned earlier. To achieve transparency and error correction, it is common to train a validator-like model with individual inference stepsCobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek, Hilton, Nakano et al. (2021), giving it the ability to evaluate the quality of the inference steps, and then collaborates with the main model to enhance its inference. Process supervision has been shown to have more powerful capabilities in the field of mathematical reasoning compared to outcome supervisionLightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever and Cobbe (2023); Luo, Liu, Liu, Phatale, Lara, Li, Shu, Zhu, Meng, Sun et al. (2024). As a result, it has also been successfully applied to OpenAI’s o1 model. However, research on the paradigm is only at a preliminary stage, and its application is limited to the mathematical domain and relies on reward models. Its potential in other domains has not been fully explored.

![](images/f63e16d503e6b2624a5a1c496eed67f828576d5bd35e702f58a2a1fd706c8267.jpg)  
Figure 1: Hierarchical scene safety cognitive structure

## 3. Hierarchical Safety Cognitive Structure

A comprehensive understanding of human knowledge structures and cognitive processes is crucial for identifying key mechanisms and information flows within human cognition. This understanding provides the foundation for supervising and guiding machine cognitive processes, thereby enabling more eficient and accurate scene safety understanding. In principle, the human cognitive process begins with bottom-up perception of sensory data, progressing to comprehension, and culminating in reasoningSarafyazd and Jazayeri (2019). Human sensory perception serves as the primary mechanism for acquiring information from environmental scenes. This perceptual process encompasses the detection and recognition of diverse entity within the environment. Sensory perception forms the foundational basis for scene comprehension, providing the requisite initial data for higher-order cognitive processes.

Inspired by this progression, our study attempts to delineate this process into distinct stages, proposing the concept of cognitive hierarchy to describe the human cognitive process in scene safety understanding, as shown in Figure 1. Specifically, the cognitive hierarchy models the bottomup process from perception to scene comprehension and reasoning. The lower levels of the hierarchical structure represent the extraction of salient features and patterns from raw perceptual inputs. Intermediate levels capture the formation of conceptual representations and semantic relationships between scene elements based on perceptual features. Finally, the higher levels of the hierarchy build upon useful information obtained from the lower stages, leveraging the underlying implicit knowledge or intuition derived from this information to perform logical reasoning about the overall safety of the scene.

## 4. SSUPL Dataset Information

Based on the above principles, we have constructed a dataset SSUPL for scene safety understanding that simulates human cognitive processes. The dataset is designed in an intuitive and structured manner to facilitate eficient assimilation by machine intelligence. Each sample of the SSUPL consists of four elements: the scene, clauses describing specific entities or relationships in the scene, the safety implications of the entities or relationships (i.e., process labels), and the holistic scene safety level (i.e., final label), as shown in Figure 2. The annotator’s operational process of constructing the remaining three elements based on the scene reflects the diferent levels in the cognitive hierarchy, respectively.

## 4.1. Data Construction Process

We manually collect images with the theme of unexpected moments online, and then utilize the caption anything modelWang, Zhang, Fei, Ge, Zheng, Tang, Li, Gao, Zhao, Shan and Zheng (2023b) and the rasa-nlu-trainer tool<sup>1</sup> to annotate text information.

Getting descriptions There are currently three main approaches to obtaining scene description data: utilizing the generative power of MLLMs, manual annotation, and based on existing image caption task datasets. However, text passages generated by MLLMs often exhibit hallucinations and contain numerous subjective descriptions. Manual annotation methods are ineficient. Most of the datasets involved in the image caption task are either sentence-level datasets that roughly describe the scene or dense captioning based datasets that do not have a coherent overall description. Even if there are additional paragraph level Stanford image-paragraph datasetKrause, Johnson, Krishna and Fei-Fei (2017), the scenes covered in the dataset are almost always low or no risk. SFT for LLMs requires high-quality of data. We therefore employ a human-machine interaction approach based on the caption anything modelWang et al. (2023b) to construct high-quality scene description texts. In this study, we primarily focus on identifying various visual attributes, including size, color, and shape.

![](images/e6fc3009091cd865599125be9a5136466037894c4c6286fc18b55b18eedeab57.jpg)  
Figure 2: An illustration of the data construction process in the SSUPL dataset. The proposed SSUPL dataset comprises 3,105 scenarios with uniformly distributed final labels across five risk levels, along with 17,464 process labels. The training, test, and validation sets contain 2,100, 705, and 300 samples respectively. The distribution of final labels in the test set is uniform.

Risk matrix After labeling all the clauses in the paragraph, a risk matrixFan, Montewka, Zhang and Han (2024) is constructed, which is then used to guide the annotator in labeling the overall safety level of the scene. Scene safety level is a prediction of the severity of incidents that may occur in the scene at some time in future, therefore, the risk matrix classifies the safety level of the scene based on two dimensions: the severity and the possibility of an accident occurring. This paper considers 5 levels of safety: no risk, low risk, medium risk, high risk, and extremely high risk.

Principles oflabeling When labeling the data, we follow the following 4 principles to align cognitive as well as human values:

• Human-centered: Measure the safety of a scene in terms of the physical safety of the humans in the scene.

• Perspectives on Justice: If the characters in the scene have diferent positions, the scene safety level is measured from the righteous side of the scene.

• No over-reasoning: In both temporal evolution and spatial extension, do not consider entities and relationships that do not appear in the paragraphs unless their presence is clearly suggested.

• Maximum risk level: Humans can understand a scene in multiple diferent chains of thought, resulting in diferent risk levels based on the risk matrix. We label the data with the maximum of these risk levels.

Moreover, given a scene, we assign it to one of the categories of pre-accident scene, during-accident scene, and post-accident scene, depending on the stage in the hazardous accident cycle.

• Pre-accident scene: There is currently no direct harm to humans in the scene, but based on the scene clues, the scene has the potential to cause harm to humans after a period of time. We determine the safety level through a complete risk matrix.

• During-accident scene: The scene has a certain accident that happened some time ago. The accident has not yet ended and continues to occur, already with some consequences. Based on the risk matrix, we determine the safety level of the scene based on the severity of the cumulative consequences that may result when the accident is over.

• Post-accident scene: An accident of some kind occurred in the scene some time ago, and the accident has ended with some consequences. We categorize the consequences into reversible consequences (e.g.,

![](images/6d72e31d6bb2ef0446686c8da45279c042378e18917ca3c54343bc79b0685665.jpg)  
(a)

![](images/9640348737e2456da6c08c2f3f706ad00a365bea471580b964b50089f12be84d.jpg)  
(b)

![](images/cf45e252b7eaa5072cd9d19ecc950c1bd63503043b4506807c06409a0217ffcf.jpg)  
(c)

![](images/d37bd5f25d2c13310c131d6664a92ea7c485bec251c5df076b14332ade5d69dd.jpg)  
(d)  
Figure 3: Statistical analysis of SSUPL dataset. (a) Word count distribution. (b) Step count distribution. (c) Top 20 verbs distribution. (d) Top 20 nouns distribution.

## Table 1

Distribution of labels in the SSUPL dataset
<table><tr><td>Label</td><td>Number</td></tr><tr><td>no risk</td><td>621</td></tr><tr><td>low risk</td><td>621</td></tr><tr><td>medium risk</td><td>621</td></tr><tr><td>high risk</td><td>621</td></tr><tr><td>extremely high risk</td><td>621</td></tr><tr><td>negative</td><td>8183</td></tr><tr><td>neutral</td><td>6517</td></tr><tr><td>positive</td><td>2164</td></tr></table>

minor and serious injuries) and irreversible consequences (e.g., death), and then determine the safety level of the scene based on the likelihood of a reversible consequence developing into an irreversible consequence and on the rows of the constructed risk matrix whose severity exceeds the reversibility threshold.

## 4.2. Dataset Statistics

For scene description paragraphs of the SSUPL dataset, the word count ranges from 14 words to 175 words, as detailed in Figure 3a. For the safety reasoning process, the count of steps ranges from 2 steps to 16 steps, as detailed in Figure 3b. The dataset contains 9608 unique words, including 1035 unique verbs and 2938 unique nouns, ensuring a diversity of entities and relationships. The top 20 verbs and top 20 nouns with the highest frequency of occurrence are shown in Figure 3c and Figure 3d, respectively.

In addition, we explore the sentence diversity of the dataset. Specifically, we calculate the average CIDEr-DVedan Lawrence Zitnick and Parikh (2015) similarity between individual sentences in the descriptive paragraph of each scene and subtract the average CIDEr-D similarity from 100 as the final diversity score. The diversity score on the SSUPL dataset reaches 78.63, which is better than the Stanford image-paragraph dataset with a diversity score of 74.22. This indicates that the SSUPL dataset has more diverse utterances, and the sentences in the paragraphs provide more information about the scene.

## 5. Inference Process of Machine Intelligence According to Safety Cognitive Hierarchy

In this section, we begin by outlining the token output process of SFT LLM in safety reasoning. Following this, we conduct a detailed analysis of information transfer within the SFT LLM from the perspective of information flow, illustrating how the underlying mechanisms align with the hierarchical cognitive process structure. Moreover, we examine the critical factors that shape the final safety prediction, providing insights into their respective influences and interactions.

It is necessary to note that despite the predominance of visual information in human perception, machine systems face fundamental challenges in directly comprehending visual experiences: significant disparities exist between machine and human visual perception mechanisms, and machines struggle to capture the subjectivity and diversity inherent in visual experiences. Converting visual information into textual representations ofers structured semantic data that bridges the gap between low-level visual features and high-level semantic concepts, while facilitating more efective integration of human prior knowledge, thereby enhancing model generalization capabilities and interpretability. Therefore, this study adopts a language-mediated modal decoupling paradigmGuo, Li, Li, Tiong, Li, Tao and Hoi (2023), utilizing a mature image captioning model to convert images into descriptive text, and then employing the LLM to perform scene safety inference based on this text. The research focuses on enhancing the reasoning capabilities of LLMs.

## <sup>,</sup>5.1. Problem Formulation

The SFT LLM is capable of performing scene safety understanding in a structured format. Given a scene $i ,$ its textual description $S _ { i }$ is provided by an image caption model and then fed into the SFT LLM. The SFT LLM then autoregressively generates $k _ { i }$ judgments regarding the influence of entities or relationships on scene safety, forming a reasoning process represented as:

$$
R _ { i } = \{ ( x _ { i , j } , y _ { i , j } ) | 1 \leq j \leq k _ { i } , j \in \mathbb { Z } , \sum _ { j = 1 } ^ { k _ { i } } x _ { i , j } = S _ { i } , y _ { i , j } \in \mathbb { P } \}
$$

![](images/f2ccc6c9492e516739d8074fd12e923465e1950440e09c72d4e3bb01a7c07ac3.jpg)  
Figure 4: Schematic of the information flow for SFT LLM to predict the safety level of a test sample.

(1)

where $x _ { i , j }$ represents the words in the �-th clause describing an entity or entity relationship, and $y _ { i , j }$ represents the corresponding process label words for $x _ { i , j }$ . The process label space ℙ is defined as:

$$
\mathbb { P } = \{ n e g a t i v e , n e u t r a l , p o s i t i v e \}\tag{2}
$$

To predict the final safety level of scene �, we define $W _ { i }$ and $F _ { i }$ as the SFT LLM prompts and the final prefix words that guide the prediction, respectively. The input to the model is formulated as:

$$
C _ { i } = W _ { i } \oplus S _ { i } \oplus T ( R _ { i } ) \oplus F _ { i }\tag{3}
$$

where the �(⋅) is a function that appends process prefix words, and ⊕ denotes concatenation. The final safety level of scene � predicted by the SFT LLM based on greedy decoding strategy is:

$$
\hat { r _ { i } } = \arg \operatorname* { m a x } _ { r _ { i } \in \mathbb { D } } \mathcal { P } ( r _ { i } | C _ { i } ; \theta )\tag{4}
$$

where � represents the parameters of the SFT LLM,  denotes the probability distribution produced by the model for safety level prediction, and � is the final label space, defined as:

$$
\mathbb { D } = \{ n o r i s k , \dots , e x t r e m e l y h i g h r i s k \}\tag{5}
$$

## 5.2. Information Flow Analysis

Building upon the prior worksWang, Li, Dai, Chen, Zhou, Meng, Zhou and Sun (2023a), this study investigates the mechanisms underlying information transfer during the reasoning process within SFT LLMs. Specifically, we analyze the key factors influencing the prediction of the final safety level from the perspective of information flow. To quantify information flow within the model, we employ saliency scores between tokens, which serve as a measure of the significance of information transfer. In classification tasks, these scores reflect the contribution of individual components to the final classification outcome. For a given scene �, the saliency score matrix $I _ { l } ^ { i }$ at the �-th layer of the SFT LLM is computed as follows:

$$
I _ { l } ^ { i } = \left| \sum _ { h } A _ { l , h } ^ { i } \odot \frac { \partial \mathcal { L } ( x ) } { \partial A _ { l , h } ^ { i } } \right| .\tag{6}
$$

where $A _ { l , h } ^ { i }$ denotes the attention weight matrix associated with the ℎ-th head in the �-th layer, and (�) represents the cross-entropy loss function. The resulting $I _ { l } ^ { i }$ is a twodimensional lower triangular matrix, where each element $I _ { l } ^ { i } ( m , n )$ characterizes the significance of the information flow from the �-th token to the �-th token.

## 5.2.1. Information Flowfrom Process Labels

Given the critical role of process labels, we first compute the mean importance of information flow from tokens in $x _ { i , j }$ and $S _ { i }$ to tokens in $y _ { i , j } ,$ denoted as $\mathcal { M } _ { i , l } ^ { x y }$ and $\mathcal { M } _ { i , l } ^ { s y }$ , respectively. Furthermore, to establish a baseline for evaluating information flow intensity, we analyze the mean importance of information flow among normal tokens, denoted as $\mathcal { M } _ { i , j } ^ { x x }$ They are formulated as:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathcal { M } _ { i , l } ^ { x y } = \frac { \sum _ { ( m , n ) \in \mathbb { C } _ { i } ^ { X } } J _ { l } ^ { i } ( m , n ) } { \left| \mathbb { C } _ { i } ^ { x y } \right| } , } \\ { \mathcal { M } _ { i , l } ^ { s y } = \frac { \sum _ { ( m , n ) \in \mathbb { C } _ { i } ^ { S y } } J _ { l } ^ { i } ( m , n ) } { \left| \mathbb { C } _ { i } ^ { x y } \right| } , } \\ { \mathcal { M } _ { i , l } ^ { x x } = \frac { \sum _ { ( m , n ) \in \mathbb { C } _ { i } ^ { X } } J _ { l } ^ { i } ( m , n ) } { \left| \mathbb { C } _ { i } ^ { x x } \right| } . } \end{array} \right. } \end{array}\tag{7}
$$

where $\mathbb { C } _ { i } ^ { x y } = \{ ( q _ { m } , q _ { n } ) | m \in y _ { i , j } , n \in x _ { i , j } , 1 \leq j \leq k _ { i } , j \in$ $\mathbb { Z } \} , \mathbb { C } _ { i } ^ { s y } = \{ ( q _ { m } , q _ { n } ) | n \in S _ { i } , m \in y _ { i , j } , 1 \leq j \leq k _ { i } , j \in \mathbb { Z } \}$ $\mathbb { C } _ { i } ^ { x x } = \{ ( q _ { m } , q _ { n } ) | m , n \in x _ { i , j } , q _ { n } < q _ { m } , 1 \leq j \leq k _ { i } , j \in \mathbb { Z } \}$ Here, $q _ { n }$ represents the position of token �.

## 5.2.2. Information Flowfor Final Safety Level Prediction

To analyze the information flow contributing to the final safety level prediction, we consider four sets of tokens: $S _ { i } ,$ $x _ { i , j } , y _ { i , j }$ , and $F _ { i } ,$ , representing diferent components of the SFT LLM input and inference process (as illustrated in Figure 4). The mean importance of the information flow from these sets to the final safety level prediction is denoted as $\mathcal { M } _ { i , l } ^ { s r } , \mathcal { M } _ { i , l } ^ { x r } , \mathcal { M } _ { i , l } ^ { y r } , \mathcal { M } _ { i , l } ^ { f r }$ , respectively. These are defined as follows:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathcal { M } _ { i , l } ^ { s r } = \frac { \sum _ { ( m , n ) \in \mathbb { S } _ { t } ^ { r } } I _ { l } ^ { i } ( m , n ) } { \left. \mathbf { C } _ { l } ^ { s r } \right. } , } \\ { \mathcal { M } _ { i , l } ^ { x r } = \frac { \sum _ { ( m , n ) \in \mathbb { S } _ { t } ^ { r } } I _ { l } ^ { i } ( m , n ) } { \left. \mathbf { C } _ { l } ^ { s r } \right. } , } \\ { \mathcal { M } _ { i , l } ^ { y r } = \frac { \sum _ { ( m , n ) \in \mathbb { S } _ { t } ^ { y r } } I _ { l } ^ { i } ( m , n ) } { \left. \mathbf { C } _ { l } ^ { s r } \right. } , } \\ { \mathcal { M } _ { i , l } ^ { f r } = \frac { \sum _ { ( m , n ) \in \mathbb { S } _ { t } ^ { r } } I _ { l } ^ { i } ( m , n ) } { \left. \mathbf { C } _ { l } ^ { s r } \right. } } \end{array} \right. } \end{array}\tag{8}
$$

where $\mathbb { C } _ { i } ^ { s r } ~ = ~ \{ ( q _ { \hat { r _ { i } } } , q _ { n } ) | n ~ \in ~ S _ { i } \} , \mathbb { C } _ { i } ^ { x r } ~ = ~ \{ ( q _ { \hat { r _ { i } } } , q _ { n } ) | n ~ \in ~ \mathbb { { \Lambda } }$ $x _ { i , j } , 1 \ \leq \ j \ \leq \ k _ { i } , j \ \in \ \mathbb { Z } \} , \mathbb { C } _ { i } ^ { y r } = \ \{ ( q _ { \hat { r _ { i } } } , q _ { n } ) | n \in \ y _ { i , j } , 1 \leq$ $j \leq k _ { i } , j \in \mathbb { Z } \} , \mathbb { C } _ { i } ^ { f r } = \{ ( q _ { \hat { r } _ { i } } , q _ { n } ) | n \in F _ { i } \}$

## 5.2.3. Experimental Analysis ofInformation Flow Mechanism on SFT LLM

To investigate the internal information flow mechanisms and their impact on final safety reasoning ability, we selected five widely used open-source models: LLama2-7B, which features 32 hidden layers from the LLama familyTouvron, Martin, Stone, Albert, Almahairi, Babaei, Bashlykov, Batra, Bhargava, Bhosale et al. (2023); GPT-J, consisting of 28 layers from the GPT familyWang and Komatsuzaki (2021); Gemma2-9B, which includes 42 layers from the Gemma familyTeam, Mesnard, Hardin, Dadashi, Bhupatiraju, Pathak, Sifre, Rivière, Kale, Love et al. (2024); Intern2.5-7B, incorporating 32 layers from the Intern familyCai, Cao, Chen, Chen, Chen, Chen, Chen, Chen, Chen, Chu et al. (2024); and Baichuan2-7B, which comprises 32 layers from the Baichuan family Yang, Xiao, Wang, Zhang, Bian, Yin, Lv, Pan, Wang, Yan et al. (2023).

We first perform SFT on the selected LLMs using the training set. The SFT LLMs are subsequently employed to conduct information flow experiments on the test set. Given hardware constraints, we utilize Low-Rank Adaptation (LoRA)Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang and Chen (2021) for SFT and integrate the resulting adapter into the base model. The LoRA rank is set to 8, the number of training epochs is set to 5, the initial learning rate is set to $5 \times 1 0 ^ { - 5 }$ , and the warmup ratio is set to 0.1. All experiments are conducted using two NVIDIA A6000 GPUs (48GB memory each) and are implemented via LLamafactoryZheng, Zhang, Zhang, Ye, Luo, Feng and Ma (2024) with our modifications. For Llama2-7B-chat, Gemma2-9Bit, and GPT-J, we designate �\_���� and �\_���� as LoRA target modules. For Baichuan2-7B-chat and Intern2.5-7Bchat, LoRA is applied to ��� modules.

![](images/d73a6525225251c7f8fd08b3ae51f19d7a624720f93f4e73ff6a1963c4c4a105.jpg)

(a)  
![](images/2c9e2c151446936c7c931f8fb6b549ebade70550b9eb7fe5a629277cc8e6efa5.jpg)  
(b)  
Figure 5: Information flow. (a) The information flow reflecting hierarchical cognitive structure in Llama2-7b-chat-sft. (b) Importance of information flow for each layer in Llama2-7bchat-sft for the prediction of final scene safety level. Results for other models can be found in Appendix A.1 and Appendix $\mathsf { A } . 2$

Figure 5a illustrates the role of information flow associated with process labels across diferent layers of the SFT LLMs, as well as the significance of information flow among normal tokens. It can be seen that $\mathcal { M } _ { x y }$ dominates in the shallow layer, accounting for about 40%, while $\mathcal { M } _ { x x } , \mathcal { M } _ { y r }$ and $\mathcal { M } _ { s r }$ account for about 30%, 20% and 10%, respectively. $\mathbf { A } \mathbf { s }$ the layer deepens, $\mathcal { M } _ { y r }$ approximates monotonically increasing and converges to 1. In contrast, the other information flows approximately monotonically decrease and converge to 0. These phenomena indicate that process labels serve as semantic anchors throughout the reasoning process. In the shallow layers, they facilitate the aggregation of information from relevant clauses and model inputs, contributing to the formation of semantic representations. In the deeper layers, the SFT LLMs extract information from these process labels, which are rich in contextual cues, to generate the final prediction. This hierarchical mechanism aligns with the hierarchical safety cognitive structure outlined in Section 3.

Figure 5b further examines the influence of input words and distinct components of the inference process on the final prediction across all layers of the SFT LLMs. It can be seen that the change of $\boldsymbol { \mathcal { M } } _ { f r }$ shows a pattern of decreasing and then increasing as the layers deepen, with a stable leadership position in the shallow and deep layers, accounting for between 60% and 80% of the total. The change of $\mathcal { M } _ { y r }$ is completely opposite, exceeding 40% in the middle layers, thus replacing $\boldsymbol { \mathcal { M } } _ { f r }$ to occupy the leadership position. In addition, the share of $\mathcal { M } _ { s r }$ and $\mathcal { M } _ { x r }$ is consistently low. In summary, the key findings are as follows:

(1) The clause component of the inference process $( x _ { i , j } )$ exhibits a functional similarity to the model input $( S _ { i } )$ , both exerting a relatively indirect influence on the final prediction.

(2) In both shallow and deep layers, prefix words that introduce the final prediction (e.g., "Final risk level:") exert the most significant impact. This pattern is analogous to the phenomenon in ICLMin, Lyu, Holtzman, Artetxe, Lewis, Hajishirzi and Zettlemoyer (2022). By applying the information flow blocking methodWang et al. (2023a), we observe that words outside the five predefined safety levels occasionally appear in the final prediction. This finding suggests that these prefix words function as category constraints, efectively narrowing the classification space and ensuring that the five safety levels consistently rank among the most probable words in the vocabulary.

(3) Compared to the normal textual elements, process labels play a particularly critical role during inference, especially in the middle layers of the model. This observation further reinforces their function as semantic anchors, as demonstrated in Figure 5a.

## 5.3. Distribution Correlation Analysis

From a cognitive perspective, the cumulative distribution of hazards within a given scene is closely correlated with the overall distribution of safety. Specifically, a scene characterized by a significant number of hazardous objects or a particularly dangerous object is likely to present an extremely high risk level. This subsection intends to translate this cognitive understanding into a reliable framework of prior knowledge within the context of machine intelligence, thereby informing and guiding subsequent research.

## 5.3.1. Distribution Formulation

In the context of this study, the intensity of a particular entity/relationship with a specific process label can be represented by the attention given to it by the final safety level. We use $A _ { l } ^ { i }$ to represent $\begin{array} { r } { \frac { 1 } { H } \sum _ { h } A _ { l , h } ^ { i } . } \end{array}$ , where � is the number of heads. $\mathbb { S } _ { i , l } ^ { 1 }$ and $\mathbb { S } _ { i , l } ^ { 2 }$ denote the attention distributions on process label tokens and normal text tokens guided by corresponding process labels when predicting the final safety level, respectively. $P _ { i }$ represents the probability distribution of the final prediction result. $\mathbb { S } _ { i , l } ^ { 1 } , \mathbb { S } _ { i , l } ^ { \hat { 2 } }$ and $P _ { i }$ are shown as follows:

$$
\mathbb { S } _ { i , l } ^ { 1 } = \left[ \sum _ { j = 1 } ^ { k _ { i } } \sum _ { \substack { t \in y _ { i , j } } } A _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) , \sum _ { j = 1 } ^ { k _ { i } } \sum _ { \substack { t \in y _ { i , j } } } A _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) , \sum _ { j = 1 } ^ { k _ { i } } \sum _ { \substack { t \in y _ { i , j } } } A _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) \right]\tag{9}
$$

$$
\mathbb { S } _ { i , l } ^ { 2 } = \left[ \sum _ { j = 1 } ^ { k _ { i } } \sum _ { { t } \in { x } _ { i , j } } A _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) , \sum _ { j = 1 } ^ { k _ { i } } \sum _ { { t } \in { x } _ { i , j } } A _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) , \sum _ { j = 1 } ^ { k _ { i } } \sum _ { { t } \in { x } _ { i , j } } A _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) \right]\tag{10}
$$

$$
P _ { i } \mathrm { = } \big [ \mathcal { P } ( r _ { i } = \mathbb { D } _ { 1 } | C _ { i } ; \theta ) , \mathcal { P } ( r _ { i } = \mathbb { D } _ { 2 } | C _ { i } ; \theta ) , \ldots , \mathcal { P } ( r _ { i } = \mathbb { D } _ { 5 } | C _ { i } ; \theta ) \big ]\tag{11}
$$

## 5.3.2. Correlation Indicator

We adopt the AUC-ROC score to measure the correlation between attention distributions and model predictions of safety levels. Specifically, the AUC-ROC score $D _ { i , l } ^ { n }$ is calculated for $\mathbb { S } _ { i . l } ^ { n } ~ \cdot ~ \mathcal { T } _ { n } + \mathcal { H } _ { n } , n ~ \in ~ \{ 1 , 2 \}$ and $\mathcal { P } _ { i } ,$ where $n ~ \in ~ \{ 1 , 2 \} . ~ \mathcal { T } _ { n } ~ \in ~ \mathbb { R } ^ { | \mathbb { P } | \times | \mathbb { D } | } , \mathcal { H } _ { n } ~ \in ~ \mathbb { R } ^ { | \mathbb { D } | }$ are the global transformation matrices, which can be additionally learned by the training samples of SFT. Further, we introduce the metric $c _ { i , l } ^ { n }$ to measure the cumulative contribution of the first � layers of the model to the prediction of the final safety levelWang et al. (2023a). It can be expressed as:

$$
C _ { i , l } ^ { n } = \frac { \sum _ { j = 1 } ^ { l } ( D _ { i , j } ^ { n } - 0 . 5 ) } { \sum _ { i = 1 } ^ { L } ( D _ { i , j } ^ { n } - 0 . 5 ) } , n \in \{ 1 , 2 \}\tag{12}
$$

where � is the number of layers of the SFT LLM.

## 5.3.3. Experimental Analysis of Correlation on SFT LLM

Here, we perform experiments on the SFT LLMs mentioned in Section 5.2.3. We define simple neural networks and train $\tau _ { n }$ and $\mathcal { H } _ { n }$ with training samples from SFT, and finally test the results on the test set of SFT. A learning rate of 0.01 is used, and the number of training epochs is set to 1000.

Figures 6a and 6b depict the correlation metric and cumulative contribution in each layer of the SFT LLMs involving the process labels and the normal text, respectively. We find that the values of $D ^ { 1 }$ and $D ^ { 2 }$ are higher than 0.7 in most layers of the SFT LLMs, indicating that both the attention distributions on process label tokens and normal text tokens are strongly correlated with the final predictions of the modelsWang et al. (2023a). Moreover, based on the values of $C ^ { 1 }$ and ${ \bar { c } } ^ { 2 }$ , we find that each layer in the SFT LLMs can make a positive contribution to the final prediction. In summary, it suggests that machine intelligence for scene safety understanding under LLM is highly aligned with human cognitive intuition, which can be regarded as a reliable prior knowledge.

![](images/b38364bc39ffa3cfa8f34cc7f81109e867a04c0dc53e6ddbf61d65271640dcb1.jpg)

(a)  
![](images/452e68fffbf70660d01b8ac5e8da70085f75551c3107fbb645e66bbd75a567f2.jpg)  
(b)  
Figure 6: Correlation analysis results. ${ \bf \Pi } ( { \sf a } ) \ : C _ { \scriptscriptstyle I } ^ { 1 }$ and $D _ { \iota } ^ { 1 }$ of diferent layers in Llama2-7b-chat-sft. (b) $c _ { \scriptscriptstyle { l } } ^ { 2 }$ and $\dot { D } _ { l } ^ { 2 }$ of diferent layers in Llama2-7b-chat-sft. Results for other models can be found in Appendix A.3 and Appendix A.4.

## 6. Process Supervision Model based on Hierarchical Cognitive Structure

Subject to the autoregressive properties of LLMs and the process properties of the linguistic modality, the direct implementation of SFT for LLMs using SSUPL in Section 5 constitutes a formal-level process supervision approach. However, this formal-level process supervision method only supervises at the formal level, emphasizing the matching of process labels without deeply modeling cognitive-level reasoning mechanisms. This means that the models may have learned process patterns superficially but lack understanding of the underlying logic and reasoning pathways. Additionally, its training approach uses end-to-end global parameter optimization for overall adjustment, making the model dificult to decompose during the inference process and consequently challenging to adapt to complex reasoning tasks.

The objective of this study is to construct a progressively modular machine model that facilitates the development of hierarchical cognitive structures through processsupervised training of modular components. This methodology is designed to enhance the machine model with humanlike reasoning and cognitive capabilities. Two fundamental challenges must be addressed to achieve this objective: (1) functional modularity: structuring the model into specialized functional components to enhance interpretability and eficiency. (2) distributed collaboration: ensuring seamless cooperation between diferent modules to optimize performance. Fortunately, the LoRA frameworkHu et al. (2021) and the MoE paradigmJiang, Sablayrolles, Roux, Mensch, Savary, Bamford, Chaplot, Casas, Hanna, Bressand et al. (2024) are well-aligned with the technical architecture of LLMs: LoRA facilitates the eficient generation of adapters, enabling functional modularity, and MoE provides a flexible mechanism for dynamic multi-module collaboration. Through functional modularity and distributed collaboration, combined with LoRA and MoE technologies, this study aims to construct a model architecture that better simulates human hierarchical cognitive structures, thereby enhancing safety assessment and decision reliability in complex environments.

## 6.1. Main Architecture of the Model

A typical model architecture for scene safety understanding and the process-supervised training framework reflecting the cognitive hierarchy model is depicted in Figure 7. The framework utilizes an existing image caption model to convert scene images into textual captions, focusing on optimizing and utilizing the reasoning capabilities of the LLM to achieve safety assessment. This process-supervised training framework comprises two critical training phases: distributed micro training and centralized macro learning.

Distributed micro training In this phase, the objective is to train a set of LoRA modules using data from singlelevel tasks, ensuring that each LoRA specializes in a specific cognitive-level task. These trained modules will subsequently serve as experts in the next stage. The training data for a single-level task incorporates the results of the preceding task as conditional inputs, as illustrated in Figure 7.

For a pre-trained module at layer �, denoted by the weight matrix $\hat { w } _ { 0 } ^ { l } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ , the LoRA technique updates the module weights by means of $\mathcal { W } _ { 0 } ^ { l } { + } \Delta \mathcal { W } ^ { l } = \mathcal { W } _ { 0 } ^ { l } { + } \mathcal { A } ^ { l } \mathcal { B } ^ { l }$ , where $\mathcal { A } ^ { l } \in \mathbb { R } ^ { d _ { 1 } \times \xi } , \mathcal { B } ^ { l } \in \mathbb { R } ^ { \xi \times d _ { 2 } } , \xi \ll m i n ( d _ { 1 } , d _ { 2 } )$ . During training, the original weights $\mathcal { W } _ { 0 } ^ { l }$ remain frozen, and only $\mathcal { A } ^ { l }$ and $B ^ { l }$ are trained, enabling eficient parameter fine-tuning with minimal computational overhead. Upon completion of this stage, a LoRA set $\Omega = \{ ( \mathcal { A } _ { i } , \boldsymbol { B } _ { i } ) \} _ { i = 1 } ^ { N }$ is obtained, where each pair $( A _ { i } , B _ { i } )$ is responsible for a specific low-level cognitive task, $\dot { \mathcal { A } _ { i } } = \{ \mathcal { A } _ { i } ^ { 1 } , \dot { \mathcal { A } _ { i } ^ { 2 } } , . . . , \mathcal { A } _ { i } ^ { L } \} , \mathcal { B } _ { i } = \{ \mathcal { B } _ { i } ^ { 1 } , \mathcal { B } _ { i } ^ { 2 } , . . . , \mathcal { B } _ { i } ^ { L } \}$ , and � is the number of tasks. In this study, based on the proposed cognitive hierarchical model, the scene safety understanding task is decomposed into three distinct cognitive tasks: perception, judgment, and cognition, corresponding to $N = 3$ For detailed LoRA training templates for these tasks, refer to Appendix B.2.

Centralized macro learning This stage aims to learn a macro-expert that can perform cognitive tasks from the lowest to the highest level of cognition and the way it collaborates with the micro-expert. Denote the macro-expert by $( A _ { 0 } , B _ { 0 } )$ . Based on the idea of $\mathrm { M o E } ,$ in this phase, the micro-experts in the $\Omega$ are first copied into the pre-trained module of the corresponding layer and their parameters are frozen, as shown in Figure 7. The input $x \in \mathbb { R } ^ { d _ { 1 } }$ outputs $x ^ { \prime } \in$ $\mathbb { R } ^ { d _ { 2 } }$ after the experts’ operations of downscaling, interaction, upscaling and collaboration, which can be flexibly designed under this framework.

Lora receiptive field: The two modules $\mathcal { A } ^ { l }$ and $B ^ { l }$ of Lora contain implications for feature extraction and extra knowledgeWu, Huang and Wei (2024b), respectively. Individual layers of a LoRA exhibit unique features, and all layers accumulate to define the overall characteristics of the LoRAWu et al. (2024b). Therefore, the architecture of this paper introduces cross-layer micro-expert modules and proposes the concept of the lora receiptive field, so as to break the limitation of same-layer correspondences and mix the learning features of each layer to enhance the capture of relevant information. Lora receptive field represents the range of the same type of Lora in other layers that a given Lora in a layer can perceive. It can be controlled collaboratively by two hyperparameters: lora radius and lora step length, using � and � to represent them respectively.

Interaction strategy: In the cognitive hierarchy, there are channels for interaction between diferent levelsJiang, Li, Chen, Zhu, Yi, Li, Zhang, Peng, Si, Cao et al. (2023). Inspired by this, we introduce interaction matrices in the latent space of experts to realize the interaction between modules of diferent experts. We use $\boldsymbol { \mathcal { I } } _ { a } ^ { b } \in \mathbb { R } ^ { \xi \times \xi }$ to denote the interaction matrix between � and �. The set of outputs of the features in the latent space after going through the interaction matrix and upscaling is:

$$
\mathcal { V } _ { l } = \{ x a T _ { a } ^ { b } b | a \in \{ \mathcal { A } _ { 0 } ^ { l } , . . . , \mathcal { A } _ { N } ^ { l } \} , b \in \mathcal { R } ^ { l } \cup \{ \mathcal { B } _ { 0 } ^ { l } \} \}\tag{13}
$$

In particular, ${ \cal T } _ { a } ^ { b }$ is a zero matrix if there is no interaction between � and $^ { b , }$ and a unit matrix if a vanilla interaction strategy is applied. $\mathcal { R } ^ { l }$ is a set controlled by the lora receptive field, denoted as:

$$
\mathcal { R } ^ { l } = \{ \mathcal { B } _ { i } ^ { j } | i \in \{ 1 , . . . , N \} , j \in \{ l - \epsilon \lambda , . . . , l , . . . , l + \epsilon \lambda \} \}\tag{14}
$$

Routing strategy: After obtaining ${ \mathcal { V } } _ { l } .$ a routing strategy is needed to assign weights to the elements in the set. In this paper, we design a routing strategy based on gating network. The macro-expert is the main goal of this phase, so the result coming exclusively from it does not need to be multiplied by a special weight. The set of weights that need to be assigned by the gating network is $\mathcal { V } _ { l } \setminus \{ x \mathcal { A } _ { 0 } ^ { l } T _ { \mathcal { A } _ { 0 } ^ { l } } ^ { { B } _ { 0 } ^ { l } } \mathcal { B } _ { 0 } ^ { l } \}$ , and we use $\mathcal { V } _ { l } ^ { * }$ to represent it. Dimensionality reduction, concatenation, normalization, and flattening need to be performed sequentially before feeding the results into the gating network. Dimensionality reduction utilises a learnable matrix $\mathcal { A } _ { r } ^ { l } \in$ $\mathbb { R } ^ { d _ { 2 } \times \mu } , \mu \ll d _ { 2 }$ to reduce the dimensionality of each result, thus avoiding an excessive training burden. Normalization aims to enhance training stabilityWu et al. (2024b). The above operation process can be described as follows:

$$
g = f _ { F } ( f _ { N } ( \bigoplus _ { v \in \mathcal { V } _ { l } ^ { * } } v \mathcal { A } _ { r } ^ { l } ) ) , g \in \mathbb { R } ^ { \left| \mathcal { V } _ { l } ^ { * } \right| \mu }\tag{15}
$$

where $f _ { N } ( . )$ and $f _ { F } ( . )$ correspond to the normalization and flattening operations, respectively. $g$ is fed into the gating network, and then the dimension is reduced to $\left| \mathcal { V } _ { l } ^ { * } \right|$ by the learnable parameter $e \in \mathbb { R } ^ { \left| \mathcal { V } _ { l } ^ { \ast } \right| \mu \times \left| \mathcal { V } _ { l } ^ { \ast } \right| }$

$$
\widetilde x = g ^ { T } \cdot e , \widetilde x \in \mathbb { R } ^ { \left| \mathcal { V } _ { l } ^ { * } \right| }\tag{16}
$$

The weight of the $i ^ { t h }$ element in $\mathcal { V } _ { l } ^ { * }$ is calculated by:

$$
\mathcal { G } ( \tilde { x } _ { i } ) = \frac { e x p ( \tilde { x } _ { i } / \tau ) } { \sum _ { j = 1 } ^ { \left| \mathcal { V } _ { l } ^ { * } \right| } e x p ( \tilde { x } _ { j } / \tau ) }\tag{17}
$$

where � is a temperature scalar that can be learnedWu et al. (2024b). Eventually, the output $x ^ { \prime }$ of the current block is:

$$
x ^ { \prime } = x \left[ \mathcal { W } _ { 0 } ^ { l } + \mathcal { A } _ { 0 } ^ { l } T _ { \mathcal { A } _ { 0 } ^ { l } } ^ { \mathcal { B } _ { 0 } ^ { l } } \mathcal { B } _ { 0 } ^ { l } \right] + \sum _ { v \in \mathcal { V } _ { l } ^ { * } } \mathcal { G } ( \tilde { x } _ { i } ) v\tag{18}
$$

In the inference phase, the model can automatically capture how each expert collaborates based on the inputs and generate the output of the current step. Both during centralized macro learning and in the inference phase, microexperts can play the role of verifier in process supervisionLightman et al. (2023), guiding training and providing interpretability.

## 6.2. Training Object

Under the proposed framework, the scene safety understanding task in this paper is decomposed into multiple subtasks and realized by multi-stage training. The quality of the training in each stage afects the final complete task, and therefore, training needs to be tailored to the task characteristics and data attributes.

Perception task For the perception task in this paper, it is essentially the segmentation of a scene description paragraph into several distinct clauses. This can be easily achieved by vanilla SFT. However, there is no single correct segmentation result for this task, and the diferences between various segmentation methods are qualitative. High-quality segmentation has a reasonably fine granularity, and each clause in the result usually contains complete semantic information, which can be helpful for the judgment task. In contrast, the semantic information in low-quality results is fragmented, and the judgment task has a high probability of arriving at neutral results based on these clauses. Therefore, the essence of this task is human preference learning in a fixed format. To achieve this, we combine the respective advantages of SFT and DPORafailov, Sharma, Mitchell, Manning, Ermon and Finn (2024) by adding the losses of both as the loss function for this task. It is calculated as follows:

![](images/55914fe6924714342e923fab570e08c0bd0a609ec488902838a5b29c89378152.jpg)  
Figure 7: Our proposed LoRA and MoE-based process-supervised training framework for scene safety understanding.

$$
\mathcal { L } _ { P } ^ { i } = \mathcal { L } - \mathbb { E } _ { \mathcal { X } _ { i } \sim \mathcal { D } } \left[ l o g \sigma ( \beta l o g \frac { \pi _ { \theta } ( X _ { i } ^ { w } | S _ { i } ) } { \pi _ { r e f } ( X _ { i } ^ { w } | S _ { i } ) } - \beta l o g \frac { \pi _ { \theta } ( X _ { i } ^ { l } | S _ { i } ) } { \pi _ { r e f } ( X _ { i } ^ { l } | S _ { i } ) } ) \right]\tag{19}
$$

where $\mathcal { X } _ { i } = ( S _ { i } , X _ { i } ^ { w } , X _ { i } ^ { l } ) , X _ { i } ^ { w }$ and $X _ { i } ^ { l }$ are the preferred and dispreferred responses, respectively, for $S _ { i }$ . � is the sigmoid function. $\beta$ is a hyperparameter. $\pi _ { \theta }$ is the policy network we need to train. $\pi _ { r e f }$ is the reference network.  is the preference dataset. We construct  for this task based on the

SSUPL, using the human annotations in the SSUPL as highquality responses and the results of random segmentation or segmentation by nouns as low-quality responses.

Judging task The task conditions the scene description paragraph and a clause to give a judgment of the efect of the clause on the overall scene safety. However, with the constructed SSUPL dataset, one faces uneven distribution of process labels. In order to avoid model overfitting to a class and model bias, we use an entropy-based regularization technique to design the loss function for this taskZhai, Likhomanenko, Littwin, Busbridge, Ramapuram, Zhang, Gu and Susskind (2023). The loss function for sample � is:

$$
\mathcal { L } _ { J } ^ { i } = \mathcal { L } - \sum _ { l = 1 } ^ { L } \sum _ { \stackrel { t \in x _ { i } } { n \in y _ { i } } } \phi ( y _ { i } ) \tilde { A } _ { l } ^ { i } ( q _ { n } , q _ { t } ) \log ( \tilde { A } _ { l } ^ { i } ( q _ { n } , q _ { t } ) )\tag{20}
$$

where $\phi ( y _ { i } )$ is a weight function of the unity formula, equal to -1 when $y _ { j } = \mathbb { P } _ { 2 }$ and 1 otherwise. $\tilde { A } _ { l } ^ { i } = \psi ( A _ { l } ^ { i } ) , \psi ( . )$ is the row-wise softmax function. The training data for this task is obtained from the SSUPL.

Table 2  
Comparison of performance under diferent methods.
<table><tr><td rowspan="2">Method</td><td colspan="7">Base Model(Accuracy% | Maximum Trainable Param.%)</td></tr><tr><td>GPT-J</td><td></td><td>Llama2-7b-chat</td><td></td><td>Baichuan2-7b-chat</td><td>Gemma2-9b-it</td><td>Intern2.5-7b-chat</td></tr><tr><td>ICL2020</td><td>12.78 | --</td><td></td><td>33.90 |--</td><td></td><td>33.05 | --</td><td>38.16 | --</td><td>41.84 | --</td></tr><tr><td>CoT2022</td><td>5.32 | --</td><td></td><td>35.18 | - -</td><td>30.64 |--</td><td></td><td>43.12 | - -</td><td>36.74 | --</td></tr><tr><td>LoRA-Final2021</td><td>68.79 | 0.0606</td><td></td><td>66.52 | 0.0622</td><td>72.34 | 0.2378</td><td></td><td>72.62 | 0.0484</td><td>72.91 | 0.2433</td></tr><tr><td>LoRA2021</td><td>65.11 | 0.0606</td><td></td><td>68.23 | 0.0622</td><td>69.79 | 0.2378</td><td></td><td>72.48 | 0.0484</td><td>73.19 | 0.2433</td></tr><tr><td>DoRA2024</td><td>64.37 | 0.0644</td><td></td><td>65.53 | 0.0661</td><td>67.52 | 0.2558</td><td></td><td>72.65 | 0.0512</td><td>71.63 | 0.2610</td></tr><tr><td>MoSLoRA2024a</td><td>63.40 | 0.0607</td><td></td><td>70.21 | 0.0623</td><td>67.94 | 0.2378</td><td></td><td>73.90 | 0.0484</td><td>72.00 | 0.2435</td></tr><tr><td>LoRAMoE2024</td><td>61.56 | 0.2571</td><td></td><td>63.40| 0.2638</td><td>67.52 | 0.9903</td><td></td><td>74.18 | 0.2062</td><td>72.48|1.0161</td></tr><tr><td>LoRA-Flow2024a</td><td>55.32 | 0.0606</td><td></td><td>61.42 | 0.0622</td><td>66.95 | 0.2378</td><td></td><td>70.21 | 0.0484</td><td>70.35 | 0.2433</td></tr><tr><td>LoRA-Hub2023</td><td>49.15 | 0.0606</td><td></td><td>52.34 | 0.0622</td><td>67.37|0.2378</td><td></td><td>70.76 | 0.0484</td><td>69.50| 0.2433</td></tr><tr><td>LoRA-Fusion2023</td><td>51.35 | 0.0606</td><td></td><td>49.79 | 0.0622</td><td>57.16 | 0.2378</td><td></td><td>66.10 | 0.0484</td><td>59.29 | 0.2433</td></tr><tr><td>SFT+ORM2023</td><td>65.82 | 0.0606</td><td></td><td>69.50 | 0.0622</td><td>71.49 | 0.2378</td><td></td><td>74.75 | 0.0484</td><td>73.90|0.2433</td></tr><tr><td>SFT+PRM2023</td><td>60.28 | 0.0606</td><td></td><td>68.23 | 0.0622</td><td>70.07| 0.2378</td><td></td><td>72.62 | 0.0484</td><td>72.77 | 0.2433</td></tr><tr><td>Ours</td><td>70.92 | 0.0722</td><td></td><td>72.49 | 0.0740</td><td>73.76 | 0.2604</td><td></td><td>76.74| 0.0598</td><td>74.61 | 0.2651</td></tr></table>

Cognitive task This task predicts the safety level of the entire scene conditional on the results of the previous two tasks. When humans measure the scene safety, they usually focus their attention mainly on the elements that have an impact on the overall safety. In the context of this paper, from an information theoretic perspective, entities/relationships with category positive or negative deserve to be more informative than those with category neutral. In addition, we know from Section 5.3.3 that the prediction of the final safety level is strongly correlated with each of the two attention distributions. Inspired by the prior knowledge, we design a loss function based on the entropy of attention distribution for this taskZhai et al. (2023). Considering the ability of process labels to aggregate information from their corresponding clauses, and computational eficiency, the loss function is designed here based on the prior from process labels as follows:

$$
\mathcal { L } _ { C } ^ { i } = \mathcal { L } - \sum _ { l = 1 } ^ { L } \sum _ { j = 1 } ^ { k _ { i } } \sum _ { t \in y _ { i , j } } \phi ( y _ { i , j } ) \tilde { A } _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) \log ( \tilde { A } _ { l } ^ { i } ( q _ { \hat { r _ { i } } } , q _ { t } ) )\tag{21}
$$

The loss function guides the model to adjust the attention distribution during training to achieve the largest possible entropy of the attention distribution for the tokens with the process labeled neutral, and the smallest possible entropy of the attention distribution for the text with the process labeled positive or negative. The training data for this task are also obtained from the SSUPL.

Centralized macro learning This phase supervised finetuning of the model with the complete scene safety understanding data, and the loss function takes the vanilla crossentropy loss.

## 6.3. Performance Evaluation

## 6.3.1. Comparison with State-of-the-art Methods

To evaluate the performance of the proposed processsupervised training method, we selected two representative approaches for comparison: one that does not require finetuning and another that involves parameter fine-tuning.

The methods that do not require finetuning include: (1) One-shot class vanilla In-Context Learning (ICL)Brown et al. (2020). For each safety level, a random sample is selected from the training data, concatenated with the corresponding safety level label, and subsequently incorporated into the prompt for the test samples. (2) The traditional chain-of-thought approach (CoT)Wei et al. (2022). The phrase ‘Let’s think step by step’ is introduced into the prompt to assist the LLMs in reasoning about the safety level of the scene.

Methods that require fine-tuning of parameters include: (1) LoRA-Final, which fine-tunes the base LLMs with data whose answers contain only the final safety level of the scene, i.e., ignoring intermediate processes in the SSUPL data. (2) LoRA, fine-tuning the base LLMs with the SSUPL dataset. (3) DoRALiu et al. (2024), which mimics the learning ability of full fine-tuning, decomposes the pre-training weights into magnitude and direction components for fine-tuning; 4) MoSLoRAWu et al. (2024a), which decomposes Lora into MoE-style subspaces and learns a mixer to fully blend the subspaces; (5) Lo-RAMoEDou et al. (2024), which adds the most typical Moe-style LoRAs to the LLM and fine-tunes all LoRAs and a router simultaneously; (6) LoRA-FlowWang et al. (2024a), which uses a fusion gate to dynamically adjust the impact of diferent LoRAs at each step of prediction; (7) LoRA-HubHuang et al. (2023), similar to LoRA-Flow, but only learns task-level fusion weights; (8) LoRA-FusionZhang et al. (2023), mixing multiple related LoRAs by fixing constant weights; (9) SFT+ORMLightman et al. (2023), which combines vanilla SFT model and outcomesupervised reward model for collaborative inference; (10)

LoRA+PRMLightman et al. (2023), which comes vanilla SFT model and process-supervised reward model for collaborative inference. Please refer to Appendix C for training details of the reward models.

Experimental settings Similar to the dataset construction pipeline, we employ the captioning anything modelWang et al. (2023b) within the framework to generate image description text. Accuracy is employed as the performance assessment metric. The rank of the LoRA involved in all methods is 8. In our approach, the parameter-frozen unit matrix is used as the interaction matrix within the interaction strategy. The descending matrix in the routing strategy is initialized using the Kaiming uniform distributionHe, Zhang, Ren and Sun (2015). The values of � and � for the LoRA receptive field are set to 1 and 3, respectively. The remaining training parameters are maintained as previously configured.

Results analysis Table 2 presents the experimental results across several approaches. From the results, we first observe that the two methods requiring no fine-tuning (ICL and CoT) perform the worst across all models, with accuracies generally below 50%. Secondly, compared with the LoRA-Final approach, the vanilla LoRA method indeed exhibits superior performance on LLama2-7b-chat (68.23% vs. 66.52%) and Intern2.5-7b-chat (73.19% vs. 72.91%), while it slightly underperforms on other SFT LLMs such as GPT-J (65.11% vs. 68.79%), Baichuan2-7b-chat (69.79% vs. 72.34%), and Gemma2-9b-it (72.48% vs. 72.62%). This performance discrepancy is tentatively attributed to exposure bias Arora, Asri, Bahuleyan and Cheung (2022), a phenomenon that will be discussed in greater detail in the subsequent section. Notably, some methods in the table (e.g., LoRAMoE, DoRA) increase the number of trainable parameters yet fail to deliver corresponding performance improvements, often performing inferior to LoRA-Final on most models. These systematic observations strongly indicate that the target task in this study is inherently challenging, and that merely increasing the parameter count is insuficient.

The experimental results delineated in Table 2 demonstrate the superior performance of our proposed method across all evaluated approaches. Our method consistently outperforms alternative fine-tuning strategies. Specifically, when compared against the second-best methods for each LLM, i.e., LoRA-Final for GPT-J and Baichuan2-7b-chat, MoSLoRA for Llama2-7b-chat, and SFT+ORM for both Gemma2-9b-it and Intern2.5-7b-chat, our framework exhibits performance improvements of 3.10%, 1.96%, 3.25%, 2.66%, and 0.96%, respectively. This consistent pattern of enhancement across diverse model architectures provides substantial empirical evidence for the generalizability and eficacy of our approach in addressing complex cognitive tasks.

Furthermore, both our method and SFT+PRM utilize process supervision. The key distinction lies in the latter’s construction of step-level preference labels $( \mathrm { i . e . , } ( x _ { i } , \mathbb { P } _ { 1 } ) \succ$ $( x _ { i } , { \mathbb P } _ { 2 } ) \ \succ \ ( x _ { i } , { \mathbb P } _ { 3 } ) )$ to evaluate process labels for intermediate inference and train reward models. However, process labels inherently possess two-sided characteristics. For instance, a fire hydrant in a scene may enhance safety by enabling firefighting during a fire, yet its presence also signifies a potential fire hazard, indirectly compromising safety. Consequently, the method of constructing process label preferences, as outlined in Appendix C, inevitably introduces some noise, which results in the performance of SFT+PRM being inferior to both our method and SFT+ORM.

Regarding training parameters, LoRAMoE requires the simultaneous training of multiple experts, resulting in the highest number of training parameters. Due to the introduction of trainable routing strategies, our framework incorporates slightly more maximum training parameters than the other methods.

## 6.3.2. Ablation Experiment

Efectiveness of lora receiptive fields Due to the flexibility and inclusiveness of this framework, we also compare the results under diferent detail settings. Specifically, we compare the performance of the LLMs in diferent settings � and �, respectively, and the results are shown in Figure 8. It can be seen that changes in both parameters can lead to changes in performance, while the former will also lead to changes in the training parameters. Specifically, for a fixed $\lambda ~ = ~ 3 .$ , the performance and training parameters show an increasing trend as � becomes larger. This is due to the gradual introduction of more experts at each layer, which increases the number of parameters that need to be trained for the routing strategy while absorbing more diverse knowledge. While in the case of fixing � = 1 and increasing �, the training parameters of the LLMs remain constant, while the performance shows a non-monotonic trend, with the optimal performance obtained at $\lambda = r o u n d ( \frac { L } { 3 } )$ . Using the index diference of the layers as a measure of the difference between the knowledge of the diferent layers, then the discrepancy function obtained for the $\epsilon = 1$ setting is $d i f f ( \lambda ) = - 1 2 \lambda ^ { 2 } + 8 L \lambda . \lambda = r o u n d ( \frac { L } { 3 } )$ allows for the widest range of LoRA receptive field within the LLMs for each layer on average, and thus allows for the uptake of the most diferentiated knowledge. Furthermore, whereas on the left and right sides of $\lambda = \bar { r } o u n d ( \frac { L } { 3 } )$ , performance roughly tends to increase and decrease, respectively, as � increases, highly consistent with the monotonicity of the discrepancy function. After � is greater than ${ \frac { L } { 3 } } .$ , the relationship between � and performance enters the next cycle.

Efectiveness of each experts To evaluate the efectiveness of each expert, we gradually introduce them in the distributed micro training phase, and the results are shown in Figure 9. As can be seen from the figure, the introduction of additional micro-experts is likely to be beneficial to the model performance. Diferent kinds of experts have diferent gain efects in the same model, and the same kind of expert has diferent enhancement efects on diferent models. With the exception of Intern2.5-7b-chat and Gemma2-9b-it, the best performance is obtained by introducing experts for the three subtasks in the rest of the LLMs. In Intern2.5-7b-chat and Gemma2-9b-it, the introduction of all experts achieves the second best performance, only slightly worse than the settings without the Risk Expert and the Aspect Expert, respectively.

![](images/f15aa8a31cfc421d7d67fde7dbaff91dde9e6ae62b5bc7eb906858dcc484e1e3.jpg)

(a)  
![](images/27045b1a3110fb402fb962dc6e45b2f92ad37f8bb3afd203bebeec3194ba3adb.jpg)  
(b)  
Figure 8: Variation of performance with diferent parameters � and �. (a) Performance varies with �. (b) Performance varies with �.

Efectiveness ofinteraction strategy The performance of the LLMs without the interaction strategy is also shown in Figure 9, as indicated by the columns filled with diagonal lines. It can be found that, except for Baichuan2-7b-chat, the interaction matrix between diferent experts is able to increase the performance of the LLMs, which indicates that the interaction between diferent experts learns useful knowledge.

## 6.3.3. The Role ofProcess Labels in Inference

Here, we explore the reasons for the performance degradation of the LoRA method based on SSUPL compared to the LoRA-Final method in the previous section, which reflects the specific role of process labels in the completion of the target task. Through Section 5.2 and Section 5.3, we have learnt that process labels plays an important role in the prediction of final safety. Here we go a step further and roughly count the accuracy of the first process label and the last process label in the inference results of SFT LLMs on the test set, as shown in Table 3. The accuracy of process label prediction is generally between 80% and 90%.

Next, we control the process labels to explore the signals provided by them for predicting the final safety level in two componentsMin et al. (2022): (i) step-process label mapping, i.e., whether $x _ { i , j }$ is mapped to the correct $y _ { i , j } ,$ (ii) process label space, i.e., $\textstyle \bigcup _ { j = 1 } ^ { k _ { i } } y _ { i , j }$ . We take a singlestep-only inference approach based on manual process label flippingWei, Wei, Tay, Tran, Webson, Lu, Chen, Liu, Huang, Zhou et al. (2023), i.e., inputting $W _ { i } \oplus S _ { i } \oplus T ( R _ { i } ) \oplus F _ { i }$ into the SFT LLMs instead of $W _ { i } \oplus S _ { i }$ . Of these, $R _ { i }$ comes from human annotation in the data annotation stage, rather than model prediction. We manually flip the $y _ { i , j }$ in the $R _ { i }$

Average prediction accuracy of the first process label and the last process label
<table><tr><td rowspan="2">Model</td><td colspan="2">Accuracy%)</td></tr><tr><td>First step</td><td>Last step</td></tr><tr><td>GPT-J</td><td>79.01</td><td>81.13</td></tr><tr><td>Llama2-7b-chat</td><td>83.12</td><td>85.82</td></tr><tr><td>Baichuan2-7b-chat</td><td>78.58</td><td>87.23</td></tr><tr><td>Gemma2-9b-it</td><td>84.54</td><td>86.38</td></tr><tr><td>Intern2.5-7b-chat</td><td>80.71</td><td>86.38</td></tr></table>

Experimental settings We consider 5 types of process label flipping: (1) Ground truth, i.e., fully correct stepprocess label mapping; (2) Positive-negative flipping (PN flipping), i.e., changing negative to positive and positive to negative based on Ground truth, and in this way get some step-process label mappings with errors; (3) All negative, i.e., set all process labels to negative. (4) All positive, i.e., set all process labels to positive. (5)All neutral, i.e., set all process labels to neutral. Note that all reasoning is performed on the test set of SSUPL. Moreover, we use the normal case, i.e., inputting $W _ { i } \oplus S _ { i }$ , as the baseline.

Results analysis Table 4 shows the results obtained by flipping the process labels under diferent SFT LLMs. It first shows that the SFT LLMs with totally correct step-process label mapping has the best performance over the normal case where there is some misclassification of process labels. This suggests that improving the correctness of the process labels is beneficial for the prediction of the final safety level. Secondly, it can be observed that positive-negative flipping does not get an accuracy that would intuitively be low. We believe that there are three reasons for this. Firstly, the impact of certain entities or relationships on scene safety is two-sided, resulting in some labels still being correct after flipping. Second, there is a high percentage of neutral in the reasoning process, making most of the process labels still correct. Finally, reasoning in this format possesses robustness, and the final safety level prediction does not depend entirely on the process labels, but also on the normal textpart, i.e., $x _ { i } , j ,$ as described in Section 5.3. In addition, the performance of all SFT LLMs is degraded when only one type of process label is present in the process label space and is particularly pronounced in the all neutral setting, suggesting that neutral provides a small amount of useful information compared to the other two types of process labels. By looking at the confusion matrix shown in Figure 10, we further find that the SFT LLMs significantly underestimate the safety level of the scenes under the all neutral setting, which indicates that the SFT LLMs are trying its best to align the prediction results with our view that ‘All neutral is safe’. In addition, under the all negative setting, the SFT LLMs hardly predict the safety level to be no risk, which is highly consistent with our perception of no risk scenes.

![](images/a5540aa5b18d770f6d46456836306682b2d5fd30f806b2ad6a6474a74e5b499c.jpg)  
Figure 9: Ablation study

Table 4  
Performance changes caused by manually flipping process labels
<table><tr><td rowspan="2">Process Label Setting</td><td colspan="5">SFT Model(Accuracy%)</td></tr><tr><td>GPT-J</td><td>Llama2-7b-chat</td><td>Baichuan2-7b-chat</td><td>Gemma2-9b-it</td><td>Intern2.5-7b-chat</td></tr><tr><td>PN flipping</td><td>59.72</td><td>65.53</td><td>68.22</td><td>66.38</td><td>65.82</td></tr><tr><td>All negative</td><td>49.36</td><td>55.18</td><td>56.45</td><td>60.00</td><td>56.31</td></tr><tr><td>All positive</td><td>52.77</td><td>58.44</td><td>64.96</td><td>63.12</td><td>51.63</td></tr><tr><td>All neutral</td><td>20.28</td><td>43.69</td><td>50.78</td><td>52.06</td><td>42.13</td></tr><tr><td>Normal case</td><td>65.11</td><td>68.23</td><td>69.79</td><td>72.48</td><td>73.19</td></tr><tr><td>Ground-truth</td><td>71.91</td><td>72.62</td><td>76.31</td><td>75.74</td><td>77.30</td></tr></table>

![](images/abc14ab7b3a0ba951e91a9f573a2defc27ec3a547756449acd6102c7659b3d71.jpg)

![](images/e90601c2673fbb1d13948014caf74d988658cbb2faffc4647ee6470776f50e49.jpg)

![](images/346d5de654196e6597df33f66edbce83821c2c96adc6e1d4e13997d092507312.jpg)

(a)  
![](images/9e163a40b899792723ae575263fc73d576ec130b8b06ba2dded3996386d978d9.jpg)  
(d)

(b)  
![](images/620820360248070b7c0295961ed74073bb77594e3b8a4b5c4b3e21e7b23923be.jpg)  
(e)

(c)  
![](images/7e2a0d3b181e338c1e6e176d3d31375dfb8207475ecc0289fa82dd411eba9f6d.jpg)  
(f)  
Figure 10: Confusion matrix for the LLama2-7b-chat-sft with diferent process labels settings. ’N’=’No risk’, ’L’=’Low risk’, ’M’=’Medium risk’, ’H’=’High risk’ and ’EH’=’Extremely high risk’ in these images. (a) PN flipping. (b) All negative. (c) All positive. (d) All neutral. (e) Normal case. (f)Ground truth.

![](images/e09b7d4819fd55880ecb54a5be9826612ee7d93041a069ce5b5d5504db9b9321.jpg)

Instruction: Please perform Scene Security Level Recognition task. Given the paragraph, tag all (aspect, impact) pairs first. Aspect should be substring of the paragraph. Impact indicates whether the corresponding aspect enhances the overall safety of the scene, it should be selected from [positive, neutral negative]. After tagging, then provide the safety level of the entire scene based on the tagging results Safety level should be selected from [no risk, low risk, medium risk, high risk, extremely high risk].

Input: A man is standing outside on a ladder that is leaning against a pole. He is working on the electrical assembly for a black street light. The man is wearing a green and gray reflective jacket, a blue top and blue pants. The back of a gray car can be seen sitting on a road below the man. The car is in front of a large brown and white building.

![](images/913fd5b42466cdab97f99b73abe23032ac1d1d8e21922fc66fb6725b86f70143.jpg)  
Figure 11: Two reasoning processes for the same scene.

Furthermore, the performance of SFT LLMs with fully correct step-process label mapping is clearly superior to the LoRA-Final method. This suggests that fully aligned inference process produces a negative alignment taxLightman et al. (2023). However, in the normal case, the inference process is not fully aligned with human due to exposure bias, and errors gradually accumulate, leading to weaker performance than the Lora-Final method in the real case. On the other hand, the LoRA-Final method outputs only a few tokens representing the final safety level, and the probability of exposure bias generation is negligible.

## 6.3.4. Interpretable Reasoning Process

Similar to PRMLightman et al. (2023), this framework also provides interpretable reasoning processes. Figure 11 illustrates an example under the Gemma2-9b-it model, where the output on the left is the correct output and the right is the incorrect output. The LoRAs obtained from distributed micro training phase act as verifiers to assess the quality or correctness of the reasoning process. Diferent LoRAs verify the results at diferent cognitive levels. Diferent colored steps are verified by diferent verifiers, the darker the color, the higher the quality. The micro-expert for the perception task normalizes the rewards of the two reasoned responses to obtain a validation score, e.g., the perceptual result on the left of the figure has a score of 0.59, while the one on the right has a score of 0.41. The micro-experts for the remaining two tasks judge the correctness of the corresponding results in terms of whether or not the results have the highest level of confidence. It can also be seen that all the steps in the output on the left are dark in color, while there are some steps in the output on the right that are light in color, which means that the verifiers have successfully identified the low quality or errors in the incorrect output.

## 7. Conclusion

In this work, we explored the possibility of integrating cognitive process modelling with process supervision by introducing, in turn, a hierarchical safety cognitive structure, the SSUPL dataset, and a process-supervised training framework that reflects the cognitive hierarchy. The training framework uses LLM as the reasoning backbone model and implements modular cognitive hierarchy functions and collaboration between diferent levels based on MoE and LoRA. The framework is flexible and scalable and outperforms various other types of approaches. The lora receptive field, interaction strategy, and routing strategy within this framework are worthy of deeper exploration in the future. This exploration represents a significant step towards bridging the gap between artificial and human intelligence in the domain of complex scene comprehension and safety assessment. We hope that the framework can be extended to other areas.

![](images/71bb64ac9e15333b41c64a71bcccaad1750420e0957ff9738ded7cf045d14dbb.jpg)  
(a)

![](images/d9b67c2af36913cba1b611e3495e1b233a64a2cb8a7b29c06e096ff4243da9a4.jpg)  
(b)

![](images/fa24923d2caaf37478a8c434b658574d6b1bc5ee7e5abdb6bab76fc2d654b4cd.jpg)

![](images/37ac2ceb67107105329cbf75e9eb46205ea3afc7998b2b436dfb855ad104d1cc.jpg)  
(c)  
(d)

Figure 12: Importance of information flow for each layer in the SFT LLMs involving process labels. (a) GPT-J. (b) Baichuan2- 7b-chat. (c) Gemma2-9b-it. (d) Intern2.5-7b-chat.  
![](images/ef6bb69bcb308062d79666715a51802382fa2e809c725c099ecf7b0ec8051110.jpg)  
(a)

![](images/cbcfe6a0df1d943a461c127d4572aa00c4df4d799fd4a4fdc40c6df062802541.jpg)  
(b)

![](images/442d2be43c0a997d5dff0328bd254e1c8b09eb35313846383b4095b05cf9a651.jpg)  
(c)

![](images/fa944f73e72d5fbb693c4a15593a61e00c0fe973ae0df15451375b87f4a962b7.jpg)  
Figure 13: Importance of information flow for each layer in the SFT LLMs for the prediction of final scene safety level. (a) GPT-J. (b) Baichuan2-7b-chat. (c) Gemma2-9b-it. (d) Intern2.5-7b-chat.  
(d)

## A. Results for Other SFT LLMs

## A.1. Results of $\mathcal { M } ^ { y r } , \mathcal { M } ^ { x r } , \mathcal { M } ^ { s r }$ and $\mathcal { M } ^ { x x }$

See Figure 12 for the $\mathcal { M } ^ { y r } , \mathcal { M } ^ { x r } , \mathcal { M } ^ { s r }$ and $\mathcal { M } ^ { x x }$ results of the remaining LLMs.

## A.2. Results of $\mathcal { M } ^ { y r } , \mathcal { M } ^ { x y } , \mathcal { M } ^ { y r }$ and $\mathcal { M } ^ { y r }$

See Figure 13 for the $\mathcal { M } ^ { y r } , \mathcal { M } ^ { x y } , \mathcal { M } ^ { y r }$ and $\mathbf { \mathcal { M } } ^ { y r }$ results of the remaining LLMs.

## A.3. Results of $C ^ { 1 }$ and $D ^ { 1 }$

See Figure 14 for the $C ^ { 1 }$ and $D ^ { 1 }$ results of the remaining LLMs.

## A.4. Results of $c ^ { 2 }$ and $D ^ { 2 }$

See Figure 15 for the $c ^ { 2 }$ and $D ^ { 2 }$ results of the remaining LLMs.

## B. Prompt Templates and Data Examples

## B.1. System Prompt Templates for Diferent LLMs

For the system prompt templates of all LLMs, we use the templates that come with LLaMA-Factory. Specifically, Llama2-7b-chat uses llama2, GPT-J uses default, Baichuan2-7b-chat uses baichuan, Gemma2-9b-it uses gemma, and Intern2.5-7b-chat uses intern2. For details, please refer to: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/ llamafactory/data/template.py

## B.3. Data Examples

For some examples of data, please refer to Table 6.

## B.2. Templates

The instruction templates for LoRA-Final, ICL, CoT and each task used in this study are displayed in Table 5.

## C. Reward Model Details

The training of reward models requires a large amount of data and building a process-supervisied dataset similar to PRM800K requires a significant amount of manpower. Thanks to the characteristics of the task labels in this study, the labeling of both the process labels and the final safety level labels contain the implicit meaning of manual sorting, $\mathrm { e . g . }$ , the preference data pairs of $( x _ { i } , \mathbb { P } _ { 1 } ) ~ \succ ~ ( x _ { i } , \mathbb { P } _ { 2 } ) ~ \succ$ $( x _ { i } , \mathbb { P } _ { 3 } )$ can be constructed for $x _ { i }$ that is manually labeled as negative, and $( C _ { i } , \mathbb { D } _ { 1 } ) \succ ( C _ { i } , \mathbb { D } _ { 2 } ) \succ ( C _ { i } , \mathbb { D } _ { 3 } ) \succ ( C _ { i } , \mathbb { D } _ { 4 } ) \succ$ $( C _ { i } , \mathbb { D } _ { 5 } )$ can be constructed for scene that is manually labeled as no risk. Based on this feature, we use only a portion of the original training data to supervise the fine-tuning of the model(for standardizing the format of response), and the remaining portion is used to construct the human preference dataset to train the reward model. The LoRA technique is also used to train the reward model, and its base model is the same as the base model of the SFT model.

![](images/d8f9525775c6b832bafdd8f4bac48c1b0991512fa5012662c8640077a6f37267.jpg)  
(a)

![](images/c926bf8a6bc475e589e15ae6dfad374d1fa44be92ef0ec8fea0bc2ae579c5174.jpg)  
(b)

![](images/98b59d60a08087cb38f1a33382f041cdaba4c364452714fc636bfc969f109408.jpg)  
(c)

![](images/14db840f4a683342b111402b1f798a44fdaf10d16f7ec5ed363956935c884fb3.jpg)  
(d)  
Figure 14: $c _ { \scriptscriptstyle l } ^ { 1 }$ and $D _ { l } ^ { 1 }$ of diferent layers in LLMs. (a) GPT-J. (b) Baichuan2-7b-chat. (c) Gemma2-9b-it. (d) Intern2.5-7b-chat.

![](images/3a4ce6feac9a2b909923ee149d1ae8bbd6a9a92725d429f3a7bde312bc763a5c.jpg)  
(a)

![](images/f47a04bfff8d875e2039c6cc1f4040c948f4a8d6952706786f275154009059f6.jpg)  
(b)

![](images/0bfc42785851cc00870b022093df0fae2e72462d4833c59f0f747f448238f079.jpg)  
(c)

![](images/427256286afccd1330fa7bb8bd2f739be0c5d30b22b38b5346220996691e3743.jpg)  
(d)  
Figure 15: $c _ { \scriptscriptstyle { l } } ^ { 2 }$ and $D _ { \phantom { } l } ^ { 2 }$ of diferent layers in LLMs. (a) GPT-J. (b) Baichuan2-7b-chat. (c) Gemma2-9b-it. (d) Intern2.5-7b-chat.

Table 5  
Instruction templates for diferent methods.
<table><tr><td>Method/Task</td><td>Instruction Template</td></tr><tr><td>CoT</td><td>Please perform Scene Safety Level Recognition task. Given a text paragraph describing the scene in detail, assign a safety level label from [No risk, Low risk, Medium risk, High risk, Extremely high risk]. Let&#x27;s think step by step. &lt;scene description&gt;</td></tr><tr><td>ICL</td><td>Scene: &lt;scene 1&gt; Final risk level: No risk; Scene: &lt;scene 2&gt; Final risk level: Low risk; Scene: &lt;scene 3&gt; Final risk level: Medium risk; Scene: &lt;scene 4&gt; Final risk level: High risk;</td></tr><tr><td>LoRA-Final</td><td>Scene: &lt;scene 5&gt; Final risk level: Extremely high risk; Scene: &lt;test scene description&gt; Final risk level: Please perform Scene Safety Level Recognition task. Given a text paragraph describing the scene</td></tr><tr><td>Perception</td><td>in detail, assign a safety level label from [No risk, Low risk, Medium risk, High risk, Extremely high risk]. Return label only without any other text. &lt;scene description&gt; Please execute the clause partitioning task. Given a scene description, divide it into multiple</td></tr><tr><td>task Judging task</td><td>clauses, each clause describing an aspect of the paragraph. &lt;scene description&gt; Please perform the impact judgment task. Given a scene description and a clause within the paragraph that describes an aspect of the paragraph. Determine the impact of this aspect on the</td></tr><tr><td>Cognitive task</td><td>overall safety of the scene and choose from [negative, positive, neutral]. Among them, negative represents weakening the safety of the scene, positive represents enhancing the safety of the scene, and neutral represents having no impact on safety or being difficult to judge. &lt;scene description&gt; &lt;aspect&gt;</td></tr></table>

<table><tr><td>Scene Description</td><td>Response (LoRA-Final)</td><td>Response (SSUPL)</td></tr><tr><td>A group of people sitting in a cafeteria style restaurant. There&#x27;s plenty of light here. A woman in a bright pink shirt is standing by the table with a purse on her shoulder. The legs of the chairs at the table are bright blue.</td><td>No risk</td><td>Aspect: A group of people sitting in a cafeteria style restaurant. Impact: Neutral; Aspect: There&#x27;s plenty of light here. Impact: Positive; Aspect: A woman in a bright pink shirt is standing by the table with a purse on her shoulder. Impact: Neutral; Aspect: The legs of the chairs at the table are bright blue. Impact: Neutral; Final Risk Level: No risk</td></tr><tr><td>People sitting inside a row boat on the water. The water is very calm with very few small waves in it. There are several other boats in the water as well. There is a bridge behind the boats above the water. There is a building next to the water.</td><td>Low risk</td><td>Aspect: People sitting inside a row boat on the water. Impact: Negative; Aspect: The water is very calm with very few small waves in it. Impact: Positive; Aspect: There are several other boats in the water as well. Impact: Negative; Aspect: There is a bridge behind the boats above the water. Impact: Neutral; Aspect: There is a building next to the water. Impact: Neutral; Final Risk Level: Low risk</td></tr><tr><td>A man rides a skateboard with red wheels on the street in front of a public transportation bus. He is wearing a helmet, a striped shirt, short pants knee pads, and shoes and socks. Behind him is a thick row of trees. The bus has blue colored sides with tinted windows and a digital display near the closed doors.</td><td>Medium risk</td><td>Aspect: A man rides a skateboard with red wheels on the street in front of a public transportation bus. Impact: Negative; Aspect: He is wearing a helmet, a striped shirt, short pants knee pads, and shoes and socks. Impact: Positive; Aspect: Behind him is a thick row of trees. Impact: Neutral; Aspect: The bus has blue colored sides with tinted windows and a digital display near the closed doors. Impact: Neutral; Final Risk Level: Medium risk</td></tr><tr><td>The cliff in the scene is really high. In the distance on the other side of the water there is more land that is lower and has sparse vegetation. The water is blue green and has a small ripple from the wind. On the cliff, there are groups of people some of which are standing and others that are sitting. There is one little boy on the edge of the high cliff that is holding a large blue umbrella. He is wearing an orange shirt, gray short, and no shoes.</td><td>High risk</td><td>Aspect: The cliff in the scene is really high. Impact: Negative; Aspect: In the distance on the other side of the water there is more land that is lower and has sparse vegetation. Impact: Neutral; Aspect: The water is blue green and has a small ripple from the wind. Impact: Neutral; Aspect: On the cliff, there are groups of people some of which are standing and others that are sitting. Impact: Negative; Aspect: There is one little boy on the edge of the high cliff that is holding a large blue umbrella. Impact: Negative; Aspect: He is wearing an orange shirt,</td></tr><tr><td>A large passenger jet prepares to land at an airport on a sunny afternoon. The airport is surrounded by water and a large white jet taxis along a runway. The distance between the two planes is very close and is gradually getting closer.</td><td>Extremely high risk</td><td>gray short, and no shoes. Impact: Neutral; Final Risk Level: High risk Aspect: A large passenger jet prepares to land at an airport. Impact: Negative; Aspect: on a sunny afternoon. Impact: Positive; Aspect: The airport is surrounded by water and a large white jet taxis along a runway. Impact: Neutral; Aspect: The distance between the two planes is very close and is gradually getting closer. Impact: Negative; Final Risk Level: Extremely high risk</td></tr></table>

Table6:ExamplesofdataintheSSUPLdatasetandusedbytheLoRA-Finalapproach.  
<sub>ene</sub> <sub>I</sub>m<sup>a</sup>

![](images/1f2e173982c6256c8c4d892896342983b5ec38fea7fde70c85bd431e12510218.jpg)

![](images/e486bb4c41108473ce90a05a58270509bdda9da4d4eeed654565da9cb3d2eef1.jpg)

![](images/8c34fbb7fb2893864fbead089b9b4ac00c1951407ef7385e7d6b05bf108d4c53.jpg)

![](images/c975dce6c3a2bf97323871b53ef3eb5f269bb7f458407f712d9822798baaae2c.jpg)

![](images/5e99cd461001cb59c208caa259db783cff637cd7c46aaa002fcbc85c4c2f62f5.jpg)

## References

An, B., Zhu, S., Panaitescu-Liess, M.A., Mummadi, C.K., Huang, F., 2024. Perceptionclip: Visual classification by inferring and conditioning on contexts, in: The Twelfth International Conference on Learning Representations.

Arora, K., Asri, L.E., Bahuleyan, H., Cheung, J.C.K., 2022. Why exposure bias matters: An imitation learning perspective of error accumulation in language generation. arXiv preprint arXiv:2204.01171 .

Baars, B.J., 2006. Global workspace theory of consciousness: Toward a cognitive neuroscience of human experience? Progress in Brain Research 150, 45–53.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al., 2020. Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901.

Cai, Z., Cao, M., Chen, H., Chen, K., Chen, K., Chen, X., Chen, X., Chen, Z., Chen, Z., Chu, P., et al., 2024. Internlm2 technical report. arXiv preprint arXiv:2403.17297 .

Chen, H., Hou, L., Wu, S., Zhang, G., Zou, Y., Moon, S., Bhuiyan, M., 2024. Augmented reality, deep learning and vision-language query system for construction worker safety. Automation in Construction 157, 105158.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al., 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168

Deng, C., Ji, X., Rainey, C., Zhang, J., Lu, W., 2020. Integrating machine learning with human knowledge. Iscience 23.

Dou, S., Zhou, E., Liu, Y., Gao, S., Shen, W., Xiong, L., Zhou, Y., Wang, X., Xi, Z., Fan, X., et al., 2024. Loramoe: Alleviating world knowledge forgetting in large language models via moe-style plugin, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1932–1945.

Fan, C., Montewka, J., Zhang, D., Han, Z., 2024. A framework for risk matrix design: A case of mass navigation risk. Accident Analysis & Prevention 199, 107515.

Guo, J., Li, J., Li, D., Tiong, A.M.H., Li, B., Tao, D., Hoi, S., 2023. From images to textual prompts: Zero-shot visual question answering with frozen large language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10867–10877.

He, K., Zhang, X., Ren, S., Sun, J., 2015. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification, in: Proceedings of the IEEE international conference on computer vision, pp. 1026–1034.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 .

Huang, C., Liu, Q., Lin, B.Y., Pang, T., Du, C., Lin, M., 2023. Lorahub: Eficient cross-task generalization via dynamic lora composition. arXiv preprint arXiv:2307.13269 .

Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., Casas, D.d.l., Hanna, E.B., Bressand, F., et al., 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088 .

Jiang, L., Li, F., Chen, Z., Zhu, B., Yi, C., Li, Y., Zhang, T., Peng, Y., Si, Y., Cao, Z., et al., 2023. Information transmission velocity-based dynamic hierarchical brain networks. NeuroImage 270, 119997.

Khaleel, M., Ahmed, A.A., Alsharif, A., 2023. Artificial intelligence in engineering. Brilliance: Research of Artificial Intelligence 3, 32–42.

Krause, J., Johnson, J., Krishna, R., Fei-Fei, L., 2017. A hierarchical approach for generating descriptive image paragraphs, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 317–325.

Kung, C.H., Yang, C.C., Pao, P.Y., Lu, S.W., Chen, P.L., Lu, H.C., Chen, Y.T., 2023. Riskbench: A scenario-based benchmark for risk identification. arXiv preprint arXiv:2312.01659 .

Li, L.L., Fang, J., Xue, J., 2024. Cognitive trafic accident anticipation. IEEE Intelligent Transportation Systems Magazine .

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., Cobbe, K., 2023. Let’s verify step by step. arXiv preprint arXiv:2305.20050 .

Liu, S.Y., Wang, C.Y., Yin, H., Molchanov, P., Wang, Y.C.F., Cheng, K.T., Chen, M.H., 2024. Dora: Weight-decomposed low-rank adaptation. arXiv preprint arXiv:2402.09353 .

Lohner, A., Compagno, F., Francis, J., Oltramari, A., 2024. Enhancing vision-language models with scene graphs for trafic accident understanding. arXiv preprint arXiv:2407.05910 .

Loo, B.P., Fan, Z., Lian, T., Zhang, F., 2023. Using computer vision and machine learning to identify bus safety risk factors. Accident Analysis & Prevention 185, 107017.

Luo, L., Liu, Y., Liu, R., Phatale, S., Lara, H., Li, Y., Shu, L., Zhu, Y., Meng, L., Sun, J., et al., 2024. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592 .

Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., Zettlemoyer, L., 2022. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837 .

Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C., 2024. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36.

Sarafyazd, M., Jazayeri, M., 2019. Hierarchical reasoning by neural circuits in the frontal cortex. Science 364, eaav8911.

Shree, V., Asfora, B., Zheng, R., Hong, S., Banfi, J., Campbell, M., 2021. Exploiting natural language for eficient risk-aware multi-robot sar planning. IEEE Robotics and Automation Letters 6, 3152–3159.

Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivière, M., Kale, M.S., Love, J., et al., 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295 .

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al., 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 .

Vedantam, R., Lawrence Zitnick, C., Parikh, D., 2015. Cider: Consensusbased image description evaluation, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4566–4575.

Wang, B., Komatsuzaki, A., 2021. Gpt-j-6b: A 6 billion parameter autoregressive language model.

Wang, H., Ping, B., Wang, S., Han, X., Chen, Y., Liu, Z., Sun, M., 2024a. Lora-flow: Dynamic lora fusion for large language models in generative tasks. arXiv preprint arXiv:2402.11455 .

Wang, J., Malawade, A.V., Zhou, J., Yu, S.Y., Al Faruque, M.A., 2024b. Rs2g: Data-driven scene-graph extraction and embedding for robust autonomous perception and scenario understanding, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 7493–7502.

Wang, L., Li, L., Dai, D., Chen, D., Zhou, H., Meng, F., Zhou, J., Sun, X., 2023a. Label words are anchors: An information flow perspective for understanding in-context learning, in: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 9840–9855.

Wang, T., Zhang, J., Fei, J., Ge, Y., Zheng, H., Tang, Y., Li, Z., Gao, M., Zhao, S., Shan, Y., Zheng, F., 2023b. Caption anything: Interactive image description with diverse multimodal controls. arXiv preprint arXiv:2305.02677 .

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al., 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837.

Wei, J., Wei, J., Tay, Y., Tran, D., Webson, A., Lu, Y., Chen, X., Liu, H., Huang, D., Zhou, D., et al., 2023. Larger language models do in-context learning diferently. arXiv preprint arXiv:2303.03846 .

Wu, T., Wang, J., Zhao, Z., Wong, N., 2024a. Mixture-of-subspaces in low-rank adaptation. arXiv preprint arXiv:2406.11909 .

Wu, X., Huang, S., Wei, F., 2024b. Mixture of lora experts. arXiv preprint arXiv:2404.13628 .

Xuehui, A., Li, Z., Zuguang, L., Chengzhi, W., Pengfei, L., Zhiwei, L., 2021. Dataset and benchmark for detecting moving objects in construction sites. Automation in Construction 122, 103482.

Yang, A., Xiao, B., Wang, B., Zhang, B., Bian, C., Yin, C., Lv, C., Pan, D., Wang, D., Yan, D., et al., 2023. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305 .

Zhai, S., Likhomanenko, T., Littwin, E., Busbridge, D., Ramapuram, J., Zhang, Y., Gu, J., Susskind, J.M., 2023. Stabilizing transformer training by preventing attention entropy collapse, in: International Conference on Machine Learning, PMLR. pp. 40770–40803.

Zhang, J., Liu, J., He, J., et al., 2023. Composing parameter-eficient modules with arithmetic operation. Advances in Neural Information Processing Systems 36, 12589–12610.

Zhang, L., Wang, J., Wang, Y., Sun, H., Zhao, X., 2022. Automatic construction site hazard identification integrating construction scene graphs with bert based domain knowledge. Automation in construction 142, 104535.

Zheng, O., Abdel-Aty, M., Wang, D., Wang, Z., Ding, S., 2023. Chatgpt is on the horizon: could a large language model be suitable for intelligent trafic safety research and applications? arXiv preprint arXiv:2303.05382 .

Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., Ma, Y., 2024. Llamafactory: Unified eficient fine-tuning of 100+ language models, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Association for Computational Linguistics, Bangkok, Thailand.

Zhu, Y., Gao, T., Fan, L., Huang, S., Edmonds, M., Liu, H., Gao, F., Zhang, C., Qi, S., Wu, Y.N., et al., 2020. Dark, beyond deep: A paradigm shift to cognitive ai with humanlike common sense. Engineering 6, 310–345.