## Graphical Abstract

Zero-shot narrative detection in social messaging Jesús M. Fraile-Hernández, Anselmo Peñas, Patrick Giedemann

## Zero-Shot Narrative Classification by LLMs

![](images/12a6370528fcb29de66f2aac8b488ed738ab5d535ade2ceec169650db389a316.jpg)  
Human-Written Descriptions

![](images/9484b5189dd9ebe1d5e8a8b453b41d48a983cc6a18c497787727b2e609578b3d.jpg)  
Automatically Generated Prompts  
Ensemble Methods (e,g, Majority Voting)

![](images/87c4fee87dbb3d07151af7f522dc91079000a1d88b5ad1f639bc05a5e36745c4.jpg)

![](images/42d89695ca290349b6045d46c8bcc5d07c504f5097d4266d6fb1ee2169d0e877.jpg)  
Model Scale Impact

![](images/2d6ef8977f2b00ab7b21959e1927fcdda45fa31517088a11631b4f28a3b83f0b.jpg)

![](images/e648bda8bc3ea647f910d27ea3b322815c24fa0a306c703e98b07fd0e730b726.jpg)

## Highlights

## Zero-shot narrative detection in social messaging

Jesús M. Fraile-Hernández, Anselmo Peñas, Patrick Giedemann

• First comprehensive zero-shot evaluation of narrative and subnarrative classification on the Dipromats and SemEval datasets.

• Systematic assessment of various prompting strategies (human-authored, automatically generated, and title-only).

• Evidence that human-curated descriptions consistently outperform other prompt types.

• Ensemble methods, particularly majority voting, significantly improve robustness and accuracy, especially for subnarrative detection.

• Larger models show superior performance and robustness, while some mid-sized models ofer a good balance of stability and computational eficiency.

• Confirm the hypothesis that LLMs’ extensive contextual knowledge allows them to interpret message communicative intentions in a zero-shot approach.

# Zero-shot narrative detection in social messaging

Jesús M. Fraile-Hernández<sup>a</sup>, Anselmo Peñas<sup>a</sup>, Patrick Giedemann<sup>b</sup>

<sup>a</sup>UNED NLP & IR Group, Universidad Nacional de Educación a Distancia, Madrid, 28040, Spain <sup>b</sup>Zurich University of Applied Sciences, Winterthur, 8400, Zurich, Switzerland

## Abstract

This study investigates the zero-shot ability of large language models (LLMs) to identify and classify hidden narratives in social messages. Our research hypothesis is that LLMs’ extensive contextual knowledge allows them to interpret messages on a deeper, pragmatic level, going beyond basic sentiment or topic analysis. Experiments on the Dipromats and SemEval datasets show that providing models with humanwritten narrative descriptions significantly improves performance, without the need of training examples. In contrast, automatically generated descriptions or the use of few examples (few-shot) often degrade accuracy due to subtle shifts in framing. The study also finds that ensemble methods, particularly majority voting, enhance robustness and that larger models perform best while also being less sensitive to prompt variations. The findings validate that LLMs can efectively detect strategic narratives in a zero-shot setting, and when combined with simple ensembling and human-written descriptions, they can rival supervised systems, ofering a scalable solution for narrative detection, specially when there is no training data for the vast majority of domains.

Keywords: Zero-shot narrative classification, Narrative identification, Internal Context, Learned context, Zero-shot, Large Language Models (LLMs), Prompt engineering

## 1. Introduction

The rapid advancements in LLMs have opened new frontiers in natural language processing, demonstrating impressive capabilities in tasks ranging from text generation to complex question answering. While much research has focused on their syntactic and semantic understanding, a critical, yet underexplored, area lies in measuring their capacity for pragmatic interpretation – specifically, their ability to discern deeper communicative intentions and the underlying narratives embedded within social messages.

Human communication is inherently complex, often extending beyond the literal meaning of words. As Sperber and Wilson’s Relevance Theory posits [1], utterances carry a presumption of optimal relevance, guiding the hearer to infer the speaker’s intended meaning by balancing cognitive efort and contextual efects. This inferential process is crucial for understanding not just what is said, but why it is said, and what strategic goals the speaker aims to achieve.

## 1.1. Problem statement

In social discourse, messages frequently serve as vehicles for particular narratives – structured accounts designed to shape perceptions, influence attitudes, or motivate actions. These narratives are not always explicitly stated but are subtly conveyed through framing, emphasis, and the strategic selection of information.

A good example can be seen in political communication about a seemingly neutral topic. Consider the following statement:

## "A new government bill proposes a tax cut."

On its own, this is just a statement of fact. However, it can be used to advance very diferent narratives.

• The "Economic Growth" Narrative. This narrative frames the tax cut as a way to benefit everyone.

– Framing: The tax cut is presented as "stimulus" or a way to "put more money back in people’s pockets."

– Emphasis: The message highlights the potential for new jobs, increased consumer spending, and a more robust economy.

– Selective information: It focuses on the benefits to small business owners and families, while downplaying any potential increase in the national debt or reduction in government services.

The underlying objective is to convince the public that the policy is a universally positive step for prosperity.

• The "Social Inequality" Narrative. This narrative frames the same tax cut as a policy that primarily benefits a select few at the expense of the many.

– Framing: The tax cut is called a "giveaway to the rich" or a "corporate handout."

– Emphasis: The message highlights how a small percentage of wealthy individuals or large corporations will see the largest financial gains.

– Selective information: It focuses on the potential cuts to public services like schools, healthcare, or infrastructure, which may be needed to ofset the lost tax revenue.

The underlying objective is to convince the public that the policy is unfair and will exacerbate social and economic divisions.

In both cases, the topic is the same (a tax cut), but the narrative is the deeper, persuasive story that shapes how the audience perceives that topic.

## 1.2. Research hypothesis

Our hypothesis is that the vast contextual knowledge and sophisticated pattern recognition abilities acquired by LLMs during their training equip them with a latent capacity to interpret social messages at this profound pragmatic level. We propose that LLMs may be capable of detecting strategic communicative intentions and, crucially, identifying the specific narratives that messages are designed to promote. This goes beyond mere sentiment analysis or topic extraction, delving into the underlying persuasive or manipulative objectives of communication.

## 1.3. Objectives

The primary goal of this research is to empirically measure the extent to which LLMs can perform the task of identifying and classifying these embedded narratives. A key challenge, and a central focus of this investigation, is to assess this capability in a zero-shot setting, meaning without providing the models with explicit examples of narratives or their classifications during the evaluation phase. By exploring LLMs inherent capacity for pragmatic inference and narrative detection, this study aims to shed light on their potential for advanced social intelligence and their applicability in domains requiring nuanced understanding of human communication, such as disinformation detection, social listening, and conflict resolution. This research will contribute to our understanding of LLM capabilities beyond surface-level linguistic processing, pushing the boundaries towards truly intelligent interpretation of human intent.

Grounded in the pragmatic nature of narrative communication, this study seeks to assess the degree to which LLMs exhibit latent competence in interpreting and classifying strategic communicative intentions embedded in social discourse. Rather than treating narratives as surface-level lexical or topical features, we approach them as implicit pragmatic constructs, often conveyed through framing, emphasis, and presupposition.

The core objective is to evaluate the zero-shot capabilities of LLMs in detecting such narratives—without reliance on annotated training data or task-specific finetuning.

To this end, our investigation is guided by the following specific goals:

1. Identify the boundaries of the zero-shot capabilities of LLMs in narrative detection. Understand whether these limitations stem from model architecture, instruction design or contextual ambiguity.

2. Benchmark diferent zero-shot approaches even with existing systems that rely on supervised learning or domain-specific fine-tuning.

3. Elucidate the trade-ofs between performance and generalisability, and to determine whether the proposed method ofers competitive accuracy while avoiding the substantial costs of data annotation and model retraining.

4. Demonstrate the robustness of zero-shot approaches across multiple thematic domains.

By pursuing these objectives, we aim to contribute to the discourse on narrative identification in low-resource settings by ofering a practical and empirically grounded alternative to conventional supervised approaches.

## 1.4. Research Questions

To consistently evaluate the capacity of LLMs to perform zero-shot narrative detection, and to explore the design decisions that may influence their performance, this study is guided by the following research questions:

• RQ1: Are concise narrative titles suficient for accurate zero-shot detection, or do models benefit significantly from extended narrative descriptions?

• RQ2: Does the use of original human-created narrative descriptions produce better results than automatically rewritten or improved versions generated by the models themselves?

– RQ2.1: Among the strategies for prompt improvement (e.g., summarisation, elaboration, or stylistic rewriting), which yields the highest performance in narrative detection tasks?

• RQ3: Does the narrative detection performance improve when using selfgenerated narrative descriptions, or is it preferable to rely on descriptions produced by more capable models, such as GPT systems?

• RQ4: How efective are ensemble strategies in improving the robustness and accuracy of zero-shot narrative detection?

• RQ5: How does the diference in model size afect performance in narrative identification?

• RQ6: Are the narrative descriptions generated by LLMs suficiently coherent, semantically accurate, and pragmatically acceptable to serve as reliable inputs for narrative detection?

## 1.5. Contributions

This work makes the following contributions to the study of zero-shot narrative detection using LLMs:

• A first comprehensive evaluation of zero-shot narrative classification across two multilingual and multidomain datasets, Dipromats and SemEval, addressing both narrative and subnarrative detection tasks.

• A systematic assessment of a wide range of prompting strategies, including original human-authored narrative descriptions, automatically generated prompts by various LLMs, and concise title-only inputs, analysing their impact on detection performance.

• Evidence that original human-curated narrative descriptions consistently outperform both title-only prompts and automatically generated alternatives, highlighting the importance of editorially consistent and contextually faithful inputs.

• An investigation of the relative efectiveness of self-generated versus GPTgenerated descriptions, revealing no significant overall advantage.

• An evaluation of ensemble methods demonstrating that majority voting ensembles substantially improve robustness and accuracy, particularly in fine-grained subnarrative detection.

• An analysis of the influence of model scale, confirming that larger LLMs not only exhibit superior overall performance but also greater robustness to prompt variation, while some mid-sized models ofer a competitive balance between stability and computational cost.

• An assessment of the semantic coherence and pragmatic acceptability of LLMgenerated narrative descriptions, emphasising that their utility depends on alignment with the annotation framework to avoid divergence from groundtruth labelling.

## 1.6. Structure of the Paper

The remainder of this paper is organised as follows. Section 2 reviews the related work. Section 3 presents the experimental setting, detailing the datasets and evaluation metrics. Section 4 describes the methodology, including narrative description generation, classifier models, classification schemes, prompting strategies, and ensemble protocols. Section 5 reports the results, covering quantitative analyses, confidence intervals, and a qualitative error study. Section 6 compares our approach with the state of the art. Section 7 presents an evaluation of how the proposed approach generalises to other languages. Section 8 provides a scalability analysis, reporting computational statistics and eficiency considerations. Finally, Sections 9 and 10 draw conclusions and outline directions for future research.

## 2. Related Work

## 2.1. Theoretical Foundations and Definitions of Narrative

A narrative is a complex and multifaceted concept, with various definitions depending on the context in which it is used. It can refer to broad ideological framings, storytelling patterns, or structured sequences of events that shape public perception and discourse. In an efort to synthesize various formulations of the concept, Dennison [2] proposed a refined definition of narratives as ”selective depictions of reality across at least two points that can include one or more causal claims, and are generally generalizable and can be applied to multiple situations, as opposed to specific stories.”

Early theoretical conceptions of narrative identify core structural elements as essential to their definition. Adam [3] defines narratives as prototypical sequences exhibiting thematic unity and a chronological progression of events involving characters. In a more formalized structure, Chatman [4] distinguishes between the story (the chain of events and their participants) and the discourse (the mode of expression, whether verbal, visual, or performative). Toolan [5] also frames narratives as perceived sequences of interconnected, non-random events anchored in time and space, involving agents such as organizations, persons, and places.

A key contribution to structuralist narrative theory is Greimas’s Actantial Model [6], which proposes six core roles: Subject, Object, Sender, Receiver, Helper, and Opponent. This model underpins a number of computational frameworks by ofering a semantic scafold for representing narrative functions. Building upon these formal definitions, Piper [7] introduces a theoretical scheme for computational narrative identification, emphasizing elements such as events, temporal anchoring, and spatial framing. In addition, the study highlights how human perceptions of narrative deviate from formal definitions, with empirical findings suggesting that readers rely not only on textual features but also on cognitive responses. These discrepancies highlight the challenges models face when encoding the complexity of human narrative comprehension.

## 2.2. Narrative Taxonomies and Analysis Frameworks

In media analysis, narratives are often studied to understand how information is framed, how it spreads, and what underlying themes emerge from large-scale text corpora. Several examples of formulations have been provided in previous taxonomies and datasets. Kotseva et al. [8] created a three-level narrative taxonomy on COVID-19 and used it to classify and analyze trends over time. Li et al. [9] focused on a flat taxonomy of anti-vax narratives, while Hughes et al. [10] presented a taxonomy of typical anti-vax narratives, organized on several common tropes and rhetorical strategies. Coan et al. [11] presented a two-level taxonomy for common instances of climate change denial in short snippets. Amanatullah et al. [12] presented a flat taxonomy of common pro-Russian narratives found in alleged pro-Kremlin influence campaigns related to the war in Ukraine.

Further synthesizing the field, Santana [13] surveys existing NLP methods for narrative extraction, covering the detection of core elements-events, agents, time, and space-across various textual domains. Importantly, the work underscores the challenges of annotation in this abstract task and links these structural elements to the temporal construction of narratives over time.

## 2.3. Computational Approaches and Challenges in Narrative Detection

Various models of narrative analysis have been proposed in computational settings, each focusing on diferent dimensions such as temporal ordering, coherence, and socio-pragmatic functions [14].

In the space of unsupervised narrative detection, Wildemann et al. [15] propose a multi-stage pipeline combining topic modeling, event detection, and event linking to automatically structure competing narratives in political discourse. Similarly, Elfes et al. [16] utilize Greimas’s Actantial Model to extract narrative roles using LLMs, encode them into embeddings, and project the results into a bidimensional space for clustering narrative structures without supervision.

On the supervised end, Levi et al. [17] constructed a manually annotated dataset spanning domains such as economics, health, and immigration, and evaluated several transformer-based models as baselines for narrative classification. The reliance on annotated corpora in this work reveals the high cost and subjectivity involved in training data acquisition for this task. In addition, Haouari et al. [18] introduce a new dataset for narrative identification in the context of UK elections. Their evaluation of GPT-4o [19] in zero-shot and low-shot environments, with and without descriptive narrative inputs, provides subtle empirical evidence of the influence of context-giving descriptions on model performance.

Recent progress in narrative detection has been supported by the introduction of new shared tasks that provide standardized benchmarks, such as SemEval 2025 Task 10 and Dipromats 2024 Task 2. In this context, Singh et al. [20] present a supervised methodology involving synthetic data generation and fine-tuning LLMs using LoRA techniques [21]. In contrast, Eljadiri et al. [22] adopt a zero-shot approach based on a multi-agent system incorporating GPT-4o and GPT-4o-mini, avoiding any training data altogether. Similarly, Fraile-Hernandez et al. [23] develop a hybrid zero-shot classifier that leverages topic detection prior to narrative classification, relying solely on the narrative title rather than contextual descriptions.

## 3. Experimentation setting

This section presents the experimental framework adopted in this study. We begin by describing the datasets employed for narrative classification, highlighting their structure. Subsequently, we detail the evaluation metrics used to assess model performance, taking into account the specificities of each dataset.

## 3.1. Datasets

Choosing the right evaluation datasets is important to ensure that our study is based on realistic and socially meaningful content. To test the ability of LLMs to detect narratives in a zero-shot setting, we needed datasets that provide clear narrative definitions and cover a range of geopolitical and thematic areas. It was also important that these datasets reflect the complexity of narrative structures and include diverse language use.

We conduct our experiments using two benchmark datasets specifically designed for narrative identification: Dipromats 2024 Task 2 [24] and SemEval-2025 Task 10

Subtask 2 [25]. Both resources are publicly available and include narrative titles as well as short descriptions, documented in oficial reports and a supporting repository.<sup>1</sup>

## Dipromats 2024 Task 2

This dataset addresses narrative detection in the political domain and is constructed in two languages, English and Spanish. It comprises tweets published by oficial diplomatic accounts from four major geopolitical actors: China, Russia, the European Union, and the United States of America. Each tweet is annotated for the presence of none, one or several predefined narratives, resulting in a multi-label classification task. Each geopolitical region includes a maximum of six diferent narrative labels.

Annotations were carried out using both the narrative titles and brief narrative descriptions provided to human annotators. The test set includes 200 tweets per language and per region, totalling 1,600 instances. Notably, the organisers of this task provided a highly limited set of training and development examples: only 102 annotated tweets, uniformly distributed with approximately four examples per label. This limitation made eficient supervised fine-tuning of the original data unfeasible, leading to the use of zero-shot, few-shot or minimally supervised approaches. To illustrate the task more concretely, we provide below an example of a tweet together with some of the narratives it supports.

## Dipromats Dataset: Example of Narrative Annotation

## Example tweet:

It is the #US, not China that takes a predatory world view. China upholds the vision of a community with a shared future for mankind and a neighborhood diplomacy of amity, sincerity, mutual benefit, and inclusiveness.

## Supported Narratives:

• CH1: The West is immoral, hostile and decadent. Tweets depict the West, primarily the US, as immoral and hostile, positioning China as a victim of their reckless behavior.

• CH2: China is a benevolent power. This narrative highlights China’s cooperative stance, emphasizing support for justice, international law, economic development in other nations, and pursuit of mutual benefits.

SemEval-2025 Task 10 Subtask 2

This dataset comes from a shared task focused on multilingual narrative detection in the domain of news media. Although it includes five languages (English, Bulgarian, Portuguese, Hindi, and Russian), our experiments are restricted to the English subset.

The subtask covers two thematic and policy-relevant areas: the Russia–Ukraine war and climate change. These topics are prolific sources of narrative-rich content across both traditional and social media, enabling the study of narrative propagation in high-impact global contexts. While climate change invokes narratives related to science, economics, and sustainability, the Russia–Ukraine war gives rise to politically and ideologically charged narratives.

Narrative annotation in this dataset follows a three-tiered hierarchical structure. The top level indicates the overarching topic (e.g., ‘Russia–Ukraine War’, ‘Climate Change’). The second level identifies the main narratives, capturing extensive interpretative frameworks, while the third level specifies sub-narratives, ofering finergrained distinctions. An Other category exists at both the topic and sub-narrative levels to accommodate content that does not align with predefined labels.

This dataset includes a total of 10 main narratives and 46 sub-narratives (36 specific and 10 labelled Other) for climate change, and 11 main narratives with 49 sub-narratives (38 specific and 11 labelled Other) for the Russia–Ukraine war. To illustrate the task more concretely, we provide below an example of a tweet together with some of the narratives it supports.

## SemEval Dataset: Example of Narrative Annotation

## Example news:

Putin says what Russia needs to do to win special operation in Ukraine Russia will win the special operation in Ukraine if the society shows consolidation and composure to the enemy, President Vladimir Putin said during a visit to the Ulan-Ude Aviation Plant on March 14, Rossiya 24 TV channel said.

Russia is not improving its geopolitical position in Ukraine. Instead, Russia is fighting "for the survival of Russian statehood, for the future development of the country and our children."

"In order to bring peace and stability closer, we, of course, need to show the consolidation and composure of our society. When the enemy sees that our society is strong, internally braced up, consolidated, then, without any doubt we will come to reach what we are striving for — both success and victory," Putin said.

"Hordes of international terrorists" new sent to the purpose to accomplish this goal, Putin said.

Afterwards, the West decided to start rehabilitating Nazism in Russia’s neighbouring states, including in Ukraine.

Nevertheless, Putin continued, Russia had long tried to build partnerships with both Western countries and Ukraine. However, after 2014, when the West contributed to the coup in Ukraine, the state of afairs changed dramatically. It was then when they started exterminating those who advocated the development of normal relations with Russia, he said.

According to Putin, Russia was forced to launch the special operation to protect the population. Western countries were hoping to break Russia quickly, but they were wrong, he said adding that Russia managed to raise its economic sovereignty since 2022.

Subscribe to Pravda.Ru Telegram channel, Facebook, RSS!

The fighting in several directions in the Kursk region continues. According to the Russian side, the Ukrainian Armed Forces are redeploying to attack in a new area

## Supported Narratives and Subnarratives:

• Russia is the Victim. Statements that portray Russia as being unfairly targeted or victimized. Look for narratives that depict Russia as sufering unjust consequences.

– Russia actions in Ukraine are only self-defence. Statements that justify Russia’s action solely as legitimate self-defence and not a deliberate action.

• Blaming the war on others rather than the invader. Statements attributing responsibility or fault to entities other than Russia in the context of Russia’s invasion of Ukraine. Look for direct or implied statements that shift blame away from Russia. Consider who is being held responsible for negative events or situations.

The West are the aggressors. Statements that shift the responsibility for the conflict and escalation to the Western block. Look for direct or implied statements that mention that this conflict was a direct consequence of actions taken by the West. Consider who is being held responsible for negative events or situations.

• Hidden plots by secret schemes of powerful groups. Statements that suggest hidden plots or secretive actions by powerful groups related to the war. Look for narratives involving clandestine activities, secret agendas, or unproven allegations. Focus on claims that lack credible evidence and suggest hidden motives.

– Other.

• Discrediting Ukraine. Statements that undermine the legitimacy, actions, or intentions of Ukraine or Ukrainians as a nation. Look for direct or implied statements that attack some aspect of the Ukrainian society.

Ukraine is associated with nazism. Accusations that Ukrainian society or government has ties to or sympathies with Nazi ideology, often referencing historical events or extremist groups. This can go with discrediting Ukrainian nation, but should be used with any mention or hint of sympathy or association with (neo-)Nazism, historical or not.

In contrast to Dipromats, this task provided a substantial number of training and development examples, 2,054 annotated documents in several languages, which allowed many participating systems to adopt supervised learning strategies. The test set in English comprises 101 news articles.

As with Dipromats, narrative titles and descriptions are available both in the oficial task documentation and through the shared repository.<sup>1</sup>

Leaderboards for both tasks are publicly available, providing reference results from participating systems and facilitating reproducibility and comparative evaluation.<sup>23</sup>

## 3.2. Evaluation Metrics

To assess model performance, we adopt the evaluation protocols defined in the original tasks for both datasets. Given the subjective and inherently interpretative nature of narrative classification, diferent evaluation strategies are applied in each case to account for ambiguity and annotation variability.

## Dipromats 2024 Task 2

In this dataset, the annotation scheme follows a ternary annotation, binary classification framework (A3C2). Annotators could assign one of three possible labels to each tweet with respect to a given narrative: yes, no, or leaning. The leaning label was introduced to capture instances where the support for a narrative was subtle, implicit, or open to interpretation. Despite the ternary nature of the annotations, the prediction task is framed as a binary classification problem, where models are expected to output either yes or no for each narrative. Evaluation metrics difer based on how the leaning cases are treated, as discussed below.

As a result, three evaluation metrics were proposed, each reflecting a diferent treatment of the leaning cases:

• F1-strict, in which leaning is treated as negative (i.e., equivalent to no).

• F1-lenient, in which leaning is interpreted as matching the system prediction, whether it is yes or no, thereby treating it as a flexible agreement with either label.

• F1-average, computed as the mean of F1-strict and F1-lenient.

In this study, although both F1-strict and F1-lenient are computed for completeness, we adopt F1-average (hereafter, F1) as the primary metric for performance evaluation. This choice reflects a balanced treatment of the ambiguous leaning cases and facilitates comparability across narrative labels.

Regarding annotation reliability, inter-annotator agreement was estimated using Cohen’s κ. The reported agreement was 0.8584 for Spanish annotations and 0.8030 for English. However, these values were obtained considering the leaning label and therefore do not directly reflect agreement on binary narrative classification. In this context, the F1-lenient metric is considered a more representative indicator of the expected annotation consistency in narrative identification.

## SemEval-2025 Task 10 Subtask 2

For this dataset, the oficial evaluation metric is the sample-based F1 averaged across documents (hereafter, F1). This metric calculates the F1-score for the predicted narrative and sub-narrative labels per instance, then averages it over the entire test set.

Inter-annotator agreement was reported using Krippendorf’s α, with a score of 0.449 for main narratives and 0.388 for sub-narratives. These values indicate moderate agreement among human annotators and highlight the challenge of reliably identifying narrative structures at diferent levels of abstraction.

## 4. Methodology

This section outlines the methodological framework employed in this study for zero-shot narrative detection. The process is structured into five interrelated steps detailed in the following subsections:

1. Narrative Description Generation.

2. Prompting Strategy Design.

3. Classifier Model Selection.

4. Classification Scheme Definition.

5. Evaluation Protocol and Setup.

## 4.1. Narrative Description Generation

Several LLMs have been used to generate refined and more informative narrative/subnarrative (n/sn) descriptions. These models have been prompted with varying levels of contextual input in order to evaluate how information richness afects the results obtained in the narrative identification task. Three input configurations have been defined:

• Title only (Gen\_Title): The LLM receives the original title of the narrative, as defined by the human annotators.

• Title + Description (Gen\_Title\_Desc): Both the narrative title and the human-authored description are provided as input.

• Title + Description + Feature Guide (Gen\_Title\_Desc\_Guide): In addition to the title and the human-authored description, the model is instructed to enrich the generated narrative description by explicitly incorporating a set of analytical dimensions. These include: social, political, and economic context; key actors involved; emotions the message seeks to elicit; inferred conclusions intended for the reader; and relevant textual features. These values are obtained directly from the LLM’s generative output and are not annotated or verified by humans.

All generation scripts, prompts, and configuration details are made publicly available through the accompanying project repository.

A diverse set of LLMs was selected for this task, spanning a range of model sizes and architectures, in order to assess the influence of model capacity and pretraining characteristics on description generation. The following models were used: gpt-4o-mini [19], calme-2.4-rys-78b [26] (a fine-tuned version of Qwen 2 78B [27]), quantized to 4 bits, Gemma-3 12B it [28], Exaone-3.5 7.8B [29], and Granite-3.3 2B [30]. This model heterogeneity enables us to compare the performance of compact, eficient generators with larger, more expressive LLMs.

## 4.2. Classifier Models

Given the abstract and high-level nature of the task, narrative classification in a zero-shot setting requires models that possess extensive world knowledge and advanced reasoning capabilities. For this reason, a selection of competitive LLMs has been employed to serve as classifiers: calme-2.4-rys-78b (4-bit quantized), Gemma-3 12B IT, Exaone-3.5 7.8B, and Granite-3.3 2B.

These models have been chosen based on two principal criteria. First, they are open-access and fully executable locally on hardware with less than 100GB of VRAM, ensuring transparency and reproducibility of results. Second, they are among the highest-performing models in their respective parameter ranges on the Massive Multitask Language Understanding (MMLU-Pro) benchmark as of March 2025. This selection enables a comparative analysis of how model scale and architectural diversity afect performance on a task as abstract and linguistically subtle as narrative detection.

The variation in model sizes—from 2B to 78B parameters—also allows for an exploration of scaling efects within the zero-shot classification paradigm, when combined with diferent prompting and narrative representation strategies.

For all models and runs, a temperature of 0.75 and top-k of 5 was consistently applied, ensuring uniform generation conditions across experiments. The maximum number of labels predicted by the LLMs was unrestricted, allowing each model to output as many labels as deemed appropriate.

## 4.3. Classification Schemes

Given the structural diferences between the two datasets, we adopt distinct classification schemes for each case.

For the Dipromats dataset, each instance is explicitly associated with a specific geopolitical region. Each region has its own set of narrative categories, and thus a region-specific multi-label classification is performed. Owing to the flat nature of the taxonomy, a single-step classification approach is suficient.

In contrast, the SemEval dataset introduces a more complex hierarchical structure spanning three levels: topics, narratives, and sub-narratives. As the primary evaluation in this task focuses on the detection of sub-narratives—from which narratives and topics can be inferred—we propose and compare three distinct classification schemes:

• All-in-One: Sub-narratives are identified in a single step, using as input the complete list of sub-narratives along with their corresponding descriptions.

• Three-Step Hierarchical: Classification is carried out through a three-step process. First, the topic is identified using only the list of possible topics as input. Second, narratives are classified using the narratives associated with the topic and their descriptions as input. Finally, for each narrative selected in the previous step, sub-narratives are identified using as input only the subnarratives and their corresponding descriptions.

• Hybrid: This strategy combines the previous two. The topic is first identified using as input the list of topics. Subsequently, sub-narratives associated with that topic are identified using as input the sub-narratives and their descriptions filtered by the predicted topic.

Figure 1 illustrates the three classification strategies. Each red box denotes a model request, while green boxes represent the outputs obtained at each stage. Additionally, illustrative examples of these outputs are provided in blue to enhance interpretability. The All-in-One approach is the most eficient in terms of the number of requests, but at the cost of requiring extensive GPU memory and managing significant contextual load. The Three-Step scheme performs a lot more requests and may accumulate errors, but has the advantage that the context that is introduced in each request is smaller. Finally, the Hybrid scheme combines the advantages of both, accumulates fewer errors than the 3-step scheme by making only one first division, makes only 2 requests and has a much smaller context than the All in One scheme.

![](images/8dff7275ac86b1740af29c9e3eb8fdad90bf97fdb4ba6e66e6075adfbb4d52ee.jpg)  
Figure 1: Overview of the three SemEval classification schemes.

## 4.4. Prompting Strategies

For each classifier model described in Section 4.2, we employed eight diferent prompting strategies that vary the $\left( \mathrm { n / s n } \right)$ descriptions used as input. These descriptions were generated according to the procedures outlined in Section 4.1. To facilitate reference throughout this work, each prompting strategy is assigned a concise abbreviation reflecting its components and source of generation:

• Title: Prompts include only the title of the (n/sn) without any accompanying description.

• OrigDesc: Prompts comprise the title alongside the original human-authored description.

• Self\_Gen\_Title: Prompts comprise the title plus a description generated by the classifier model itself based solely on the title (Gen\_Title).

• GPT\_Gen\_Title: Prompts include the title and a description generated by the GPT model from the title only.

• Self\_Gen\_Title\_Desc: Prompts include the title and a description generated by the classifier model itself from both the title and original description (Gen\_Title\_Desc).

• GPT\_Gen\_Title\_Desc: Prompts include the title and a description generated by the GPT model from both the title and original description.

• Self\_Gen\_Title\_Desc\_Guide: Prompts consist of the title and a description generated by the classifier model itself based on the title, original description, and additional feature guide prompts (Gen\_Title\_Desc\_Guide).

• GPT\_Gen\_Title\_Desc\_Guide: Prompts include the title and a description generated by the GPT model incorporating the title, original description, and the feature guide prompts.

By systematically applying this comprehensive suite of prompting strategies, we are able to rigorously evaluate the impact of varying levels of contextual and descriptive richness on zero-shot narrative detection performance. This approach allows us to dissect the contribution of each description type, from minimal only title prompts to richly guided narrative representations, and assess how diferent models respond to these input variations.

The final prompt structure used across experiments follows the general design presented in Appendix A, with modifications adapted to each specific prompting strategy. All scripts used for model inference, including prompt definitions and hyperparameter configurations, are available in the public project repository accompanying this work.

## 4.5. Evaluation Protocol and Ensemble Strategies

To ensure the reliability and robustness of the results, we execute each combination of classifier model and prompting strategy across five independent runs. All runs are performed using identical hyperparameter configurations and input conditions. By averaging the evaluation metric across these five executions, we mitigate the efect of stochastic variance and obtain a more stable estimation of model performance. This averaged score is taken as the definitive performance value for each specific model–prompt pairing.

In addition to the individual evaluations, we incorporate two ensemble strategies across the five runs of each combination:

• Majority Voting Ensemble: For each instance and each possible label, we consider its occurrence across the five runs. A label is included in the final prediction if it is predicted in at least three out of five runs (i.e., strict majority). This approach emphasises consensus and mitigates outlier decisions from individual runs.

• Inclusion Ensemble: A more inclusive approach, where all labels predicted in any of the five runs are aggregated to form the final prediction. This strategy prioritises coverage and recall, potentially at the cost of increased false positives.

These ensemble techniques enable us to analyse the extent to which prediction variance afects final outputs and to explore whether aggregation can yield more reliable zero-shot classifications. Furthermore, to assess the benefits of model diversity, we construct cross-model ensembles for the two prompting strategies that exhibit the best overall performance. Specifically, we combine predictions from the four different classifier models, each executed over five runs (resulting in ensembles based on 4 × 5 predictions per instance), and apply both majority voting and inclusion strategies. This design facilitates a comprehensive investigation into the trade-ofs between model-scale diversity, prompt informativeness, and ensemble robustness.

## 5. Results

This section reports the results obtained for the narrative detection task under a zero-shot setting across the two datasets: Dipromats and SemEval. Each result corresponds to a specific combination of classifier model and prompting strategy, as defined in Section 4, with performance computed as the average over five independent runs.

For the Dipromats dataset, Table 1 presents the macro F1 scores and std for each model–prompt combination, broken down by language (Spanish and English) and classifier (Calme, Gemma, Exaone, Granite). The rows are grouped by the type of narrative input used: from original narrative titles (Title) and descriptions (OrigDesc) to diferent self-generated and GPT-generated narrative formulations. Each block includes the base classification result, calculated as the average macro F1 score across five independent runs. The subsequent two rows show the performance of ensemble methods using majority voting and inclusion based merging, respectively.

For the SemEval dataset, results are presented separately for each of the three classification schemes described in Section 4.3. Specifically, Table 4 reports the results for the All-in-One approach, Table 2 for the Three-Step Hierarchical strategy, and Table 3 for the Hybrid variant. Each table includes the macro F1 scores averaged across five runs and std for every classifier–prompt pairing. Ensemble results using both aggregation strategies are also included. For each configuration, the first column reports the F1 score for narrative prediction, while the second column reports the F1 score for subnarrative detection.

In Table 4, classifier–prompt configurations marked with a red cross (×) indicate cases where inference could not be completed due to prompt length exceeding the 100 GB VRAM available on the target hardware. These entries are omitted from comparative analysis but are included for completeness.

Table 5 presents the ensemble results obtained using the two best-performing prompting strategies overall: Title and OriginalDesc. For each strategy, we aggregate predictions across all five runs from each classifier model. The column labelled All corresponds to the ensemble that combines outputs from all four classifier models, resulting in a total of 4 × 5 = 20 runs per instance. In contrast, the column No Granite reports ensemble results excluding the Granite-3.3 2B model. This exclusion is motivated by the substantial architectural and capacity diferences of Granite, a considerably smaller LLM compared to the other three models. In the case of majority-voting ensembles, labels are included if they appear in at least 11 out of 20 predictions when using all models (All), or in at least 8 out of 15 predictions when using all models except Granite (No Granite).

## 5.1. Results Analysis

To facilitate the interpretation of the extensive set of results, this section is structured around the six research questions posed in the study. Unless otherwise indicated, values reported for the Dipromats dataset represent the average of F1 scores across Spanish and English, and for SemEval, across both narrative and subnarrative levels. All detailed calculations are available in a publicly shared spreadsheet within the supplementary repository.

RQ1: Are concise narrative titles suficient for accurate zero-shot detection, or do models benefit significantly from extended narrative descriptions?

To address this research question, we analyse the performance diferences between each prompting strategy and the baseline configuration using only narrative titles. For the Dipromats dataset, we compute these diferences across all classifier–prompt combinations in both Spanish and English settings, as presented in Table 1. For the SemEval dataset, we perform the same comparison across the two classification targets (narratives and sub-narratives) and for each classification scheme individually (Tables 4, 2, and 3). Specifically, for each configuration, we calculate the delta in macro F1 score between a given prompting approach and the Title baseline, and then average these deltas to obtain a general performance estimate for each strategy.

<table><tr><td></td><td>es Calme</td><td>en</td><td>es en Gemma</td><td>es</td><td>en Exaone</td><td>es</td><td>en Granite</td></tr><tr><td>Title</td><td>0.6345 0.5834</td><td>0.6125</td><td>0.5647</td><td>0.5519</td><td>0.4990</td><td>0.3336</td><td>0.3675</td></tr><tr><td>std</td><td>0.0032</td><td>0.0041</td><td>0.0030 0.0015</td><td>0.0083</td><td>0.0035</td><td>0.0139</td><td>0.0120</td></tr><tr><td>majority</td><td>0.6395</td><td>0.5869 0.6141</td><td>0.5675</td><td>0.5624</td><td>0.5152</td><td>0.3517</td><td>0.3773</td></tr><tr><td>inclusion</td><td>0.6154</td><td>0.5676 0.6082</td><td>0.5581</td><td>0.5242</td><td>0.4706</td><td>0.3467</td><td>0.3811</td></tr><tr><td>OrigDesc</td><td>0.6387</td><td>0.6135 0.6242</td><td>0.5932</td><td>0.5725</td><td>0.5348</td><td>0.3352</td><td>0.3526</td></tr><tr><td>std</td><td>0.0047</td><td>0.0023 0.0037</td><td>0.0020</td><td>0.0050</td><td>0.0053</td><td>0.0112</td><td>0.0056</td></tr><tr><td>majority</td><td>0.6396</td><td>0.6150</td><td>0.6266 0.5937</td><td>0.5895</td><td>0.5496</td><td>0.3351</td><td>0.3590</td></tr><tr><td>inclusion</td><td>0.6216</td><td>0.5934 0.6251</td><td>0.5855</td><td>0.5488</td><td>0.5056</td><td>0.3557</td><td>0.3683</td></tr><tr><td>Self Gen Title</td><td>0.6210</td><td>0.5714</td><td>0.5762 0.5313</td><td>0.5062</td><td>0.4837</td><td>0.2861</td><td>0.2984</td></tr><tr><td>std</td><td>0.0048</td><td>0.0017</td><td>0.0025 0.0025</td><td>0.0050</td><td>0.0076</td><td>0.0102</td><td>0.0064</td></tr><tr><td>majority</td><td>0.6249</td><td>0.5754</td><td>0.5771 0.5330</td><td>0.5374</td><td>0.5019</td><td>0.2761</td><td>0.2925</td></tr><tr><td>inclusion</td><td>0.5991</td><td>0.5561</td><td>0.5660 0.5213</td><td>0.4759</td><td>0.4587</td><td>0.3196</td><td>0.3352</td></tr><tr><td>Self Gen</td><td></td><td></td><td></td><td>0.5463</td><td>0.5120</td><td></td><td></td></tr><tr><td>Title Desc std</td><td>0.6318</td><td>0.6182</td><td>0.5864</td><td>0.5699</td><td></td><td>0.2889</td><td>0.3295</td></tr><tr><td>majority</td><td>0.0062 0.6339</td><td>0.0017 0.0020 0.6203 0.5875</td><td>0.0022</td><td>0.0093 0.5746</td><td>0.0059 0.5367</td><td>0.0102</td><td>0.0111</td></tr><tr><td>inclusion</td><td>0.6191</td><td>0.6098 0.5760</td><td>0.5721 0.5594</td><td>0.5057</td><td>0.4771</td><td>0.2799 0.3271</td><td>0.3386 0.3493</td></tr><tr><td>Self Gen</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Title Desc Guide std</td><td>0.6324</td><td>0.6062</td><td>0.5351 0.4778</td><td>0.4574</td><td>0.4275</td><td>0.2454</td><td>0.2827</td></tr><tr><td>majority</td><td>0.0052</td><td>0.0060 0.0017</td><td>0.0029</td><td>0.0167</td><td>0.0089</td><td>0.0160</td><td>0.0036</td></tr><tr><td>inclusion</td><td>0.6322</td><td>0.6084 0.5341</td><td>0.4757</td><td>0.5002</td><td>0.4554</td><td>0.2175</td><td>0.2760</td></tr><tr><td>Gen</td><td>0.6237</td><td>0.5939 0.5292</td><td>0.4839</td><td>0.4208</td><td>0.4067</td><td>0.3046</td><td>0.3228</td></tr><tr><td>GPT Title</td><td>0.6085</td><td>0.5657</td><td>0.5839 0.5237</td><td>0.5249</td><td>0.4907</td><td>0.2673</td><td>0.2979</td></tr><tr><td>std</td><td>0.0047</td><td>0.0049</td><td>0.0015 0.0027</td><td>0.0069</td><td>0.0055</td><td>0.0093</td><td>0.0093</td></tr><tr><td>majority</td><td>0.6111</td><td>0.5729 0.5827</td><td>0.5242</td><td>0.5585</td><td>0.5177</td><td>0.2569</td><td>0.2872</td></tr><tr><td>inclusion</td><td>0.5928</td><td>0.5462 0.5758</td><td>0.5162</td><td>0.4844</td><td>0.4629</td><td>0.3218</td><td>0.3327</td></tr><tr><td>GPT Gen Title Desc</td><td>0.6403</td><td>0.6135</td><td>0.6138 0.5787</td><td>0.5458</td><td>0.5243</td><td>0.2848</td><td>0.3008</td></tr><tr><td>std</td><td>0.0047</td><td>0.0051</td><td>0.0021 0.0019</td><td>0.0061</td><td>0.0042</td><td>0.0107</td><td>0.0134</td></tr><tr><td>majority</td><td>0.6415</td><td>0.6169</td><td>0.6138 0.5783</td><td>0.5665</td><td>0.5531</td><td>0.2895</td><td>0.2972</td></tr><tr><td>inclusion</td><td>0.6310 0.6006</td><td>0.6034</td><td>0.5693</td><td>0.5047</td><td>0.4734</td><td>0.3214</td><td>0.3379</td></tr><tr><td>GPT Gen</td><td>0.6266</td><td>0.6048 0.2747</td><td>0.3802</td><td>0.4846</td><td>0.4378</td><td>0.2206</td><td>0.2717</td></tr><tr><td>Title Desc Guide std</td><td>0.0054</td><td>0.0071</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>majority</td><td>0.6370</td><td>0.6134</td><td>0.0060 0.0033 0.2756 0.3852</td><td>0.0078 0.5257</td><td>0.0077 0.4614</td><td>0.0068 0.2055</td><td>0.0108 0.2623</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>inclusion</td><td>0.6117 0.5931</td><td>0.3085</td><td>0.3926</td><td>0.4487</td><td>0.4258</td><td>0.2761</td><td>0.3239</td></tr></table>

Table 1: Dipromats results: macro F1 scores and std by model, prompting strategy, and language.

<table><tr><td rowspan="2">Title</td><td colspan="2">Calme</td><td colspan="2">Gemma</td><td colspan="2">Exaone</td><td colspan="2">Granite</td></tr><tr><td>0.4296</td><td>0.2824</td><td>0.4004</td><td>0.2640</td><td>0.3482</td><td>0.2128</td><td>0.1792</td><td>0.1202</td></tr><tr><td>std</td><td>0.0227</td><td>0.0194</td><td>0.0059</td><td>0.0045</td><td>0.0072</td><td>0.0109</td><td>0.0168</td><td>0.0080</td></tr><tr><td>majority</td><td>0.4420</td><td>0.2860</td><td>0.4010</td><td>0.2660</td><td>0.3720</td><td>0.2290</td><td>0.1970</td><td>0.1480</td></tr><tr><td>inclusion</td><td>0.4560</td><td>0.3000</td><td>0.3960</td><td>0.2580</td><td>0.3510</td><td>0.2170</td><td>0.2120</td><td>0.1330</td></tr><tr><td>OrigDesc</td><td>0.4884</td><td>0.3098</td><td>0.4214</td><td>0.2862</td><td>0.3434</td><td>0.2124</td><td>0.1666</td><td>0.1056</td></tr><tr><td>std</td><td>0.0161</td><td>0.0149</td><td>0.0079</td><td>0.0081</td><td>0.0174</td><td>0.0133</td><td>0.0234</td><td>0.0134</td></tr><tr><td>majority</td><td>0.4980</td><td>0.3080</td><td>0.4360</td><td>0.3000</td><td>0.3600</td><td>0.2130</td><td>0.1960</td><td>0.1420</td></tr><tr><td>inclusion</td><td>0.5080</td><td>0.3250</td><td>0.4230</td><td>0.2820</td><td>0.3740</td><td>0.2420</td><td>0.2030</td><td>0.1180</td></tr><tr><td>Self Gen Title</td><td>0.4362</td><td>0.3042</td><td>0.3392</td><td>0.2208</td><td>0.3016</td><td>0.1792</td><td>0.1424</td><td>0.0802</td></tr><tr><td>std</td><td>0.0231</td><td>0.0224</td><td>0.0068</td><td>0.0054</td><td>0.0172</td><td>0.0135</td><td>0.0142</td><td>0.0145</td></tr><tr><td>majority</td><td>0.4490</td><td>0.3230</td><td>0.3460</td><td>0.2260</td><td>0.2950</td><td>0.1780</td><td>0.1500</td><td>0.1090</td></tr><tr><td>inclusion</td><td>0.4990</td><td>0.3500</td><td>0.3440</td><td>0.2140</td><td>0.3570</td><td>0.2330</td><td>0.1900</td><td>0.1080</td></tr><tr><td>Self Gen Title Desc</td><td>0.4852</td><td>0.3286</td><td>0.3592</td><td>0.2508</td><td>0.3106</td><td>0.1764</td><td>0.1246</td><td>0.0576</td></tr><tr><td>std</td><td>0.0237</td><td>0.0179</td><td>0.0150</td><td>0.0133</td><td>0.0226</td><td>0.0188</td><td>0.0172</td><td>0.0218</td></tr><tr><td>majority</td><td>0.5160</td><td>0.3560</td><td>0.3560</td><td>0.2510</td><td>0.2770</td><td>0.1570</td><td>0.1740</td><td>0.1350</td></tr><tr><td>inclusion</td><td>0.5040</td><td>0.3520</td><td>0.3740</td><td>0.2500</td><td>0.3990</td><td>0.2270</td><td>0.1740</td><td>0.0770</td></tr><tr><td>Self Gen Title Desc Guide</td><td>0.4522</td><td>0.3188</td><td>0.2810</td><td>0.2810</td><td>0.2836</td><td>0.1890</td><td>0.1028</td><td>0.0496</td></tr><tr><td>std</td><td>0.0184</td><td>0.0204</td><td>0.0055</td><td>0.0055</td><td>0.0241</td><td>0.0132</td><td>0.0073</td><td>0.0061</td></tr><tr><td>majority</td><td>0.4550</td><td>0.3240</td><td>0.2770</td><td>0.2770</td><td>0.2760</td><td>0.1990</td><td>0.1780</td><td>0.1430</td></tr><tr><td>inclusion</td><td>0.5080</td><td>0.3470</td><td>0.2820</td><td>0.2810</td><td>0.3600</td><td>0.2340</td><td>0.1610</td><td>0.0770</td></tr><tr><td>GPT Gen Title</td><td>0.4442</td><td>0.2810</td><td>0.3556</td><td>0.2350</td><td>0.2892</td><td>0.1600</td><td>0.1274</td><td>0.0496</td></tr><tr><td>std</td><td>0.0202</td><td>0.0200</td><td>0.0103</td><td>0.0083</td><td>0.0118</td><td>0.0199</td><td>0.0208</td><td>0.0054</td></tr><tr><td>majority</td><td>0.4590</td><td>0.2810</td><td>0.3630</td><td>0.2370</td><td>0.2780</td><td>0.1500</td><td>0.1540</td><td>0.1130</td></tr><tr><td>inclusion</td><td>0.4940</td><td>0.3330</td><td>0.3540</td><td>0.2270</td><td>0.3520</td><td>0.2120</td><td>0.1790</td><td>0.0780</td></tr><tr><td>GPT Gen</td><td>0.4684</td><td>0.3222</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Title Desc std</td><td></td><td></td><td>0.3830</td><td>0.2676</td><td>0.3124</td><td>0.1770</td><td>0.1178</td><td>0.0406</td></tr><tr><td>majority</td><td>0.0248</td><td>0.0269</td><td>0.0186</td><td>0.0186</td><td>0.0228</td><td>0.0236</td><td>0.0107</td><td>0.0090</td></tr><tr><td>inclusion</td><td>0.4750</td><td>0.3250</td><td>0.3760</td><td>0.2640</td><td>0.3280</td><td>0.1850</td><td>0.1860</td><td>0.1280</td></tr><tr><td>GPT Gen</td><td>0.5170</td><td>0.3630</td><td>0.3830</td><td>0.2630</td><td>0.3550</td><td>0.2190</td><td>0.1650</td><td>0.0690</td></tr><tr><td>Title Desc Guide</td><td>0.4530</td><td>0.3236</td><td>0.2978</td><td>0.2114</td><td>0.2526</td><td>0.1630</td><td>0.1008</td><td>0.0340</td></tr><tr><td>std</td><td>0.0210</td><td>0.0135</td><td>0.0189</td><td>0.0130</td><td>0.0146</td><td>0.0123</td><td>0.0137</td><td>0.0066</td></tr><tr><td>majority</td><td>0.4540</td><td>0.3300</td><td>0.3070</td><td>0.2200</td><td>0.2700</td><td>0.1780</td><td>0.1090</td><td>0.0920</td></tr><tr><td>inclusion</td><td>0.4910</td><td>0.3480</td><td>0.3230</td><td>0.2240</td><td>0.3180</td><td>0.2020</td><td>0.1570</td><td>0.0590</td></tr><tr><td>std</td><td>0.0145</td><td>0.0265</td><td>0.0061</td><td>0.0048</td><td>0.0108</td><td>0.0046</td><td>0.0114</td><td>0.0176</td></tr><tr><td>majority</td><td>0.5320</td><td>0.3400</td><td>0.2950</td><td>0.2940</td><td>0.3570</td><td>0.1470</td><td>0.2690</td><td>0.2080</td></tr><tr><td>inclusion</td><td>0.5380</td><td>0.3380</td><td>0.3070</td><td>0.2970</td><td>0.3830</td><td>0.1770</td><td>0.2490</td><td>0.1300</td></tr><tr><td>OrigDesc</td><td>0.4858</td><td>0.3018</td><td>0.4278</td><td>0.3242</td><td>0.3398</td><td>0.1402</td><td>0.1770</td><td>0.0800</td></tr><tr><td>std</td><td>0.0231</td><td>0.0145</td><td>0.0275</td><td>0.0210</td><td>0.0106</td><td>0.0036</td><td>0.0123</td><td>0.0138</td></tr><tr><td>majority</td><td>0.1050</td><td>0.0360</td><td>0.4220</td><td>0.3230</td><td>0.3590</td><td>0.1550</td><td>0.2070</td><td>0.1420</td></tr><tr><td>inclusion</td><td>0.3400</td><td>0.1950</td><td>0.4730</td><td>0.3390</td><td>0.3390</td><td>0.1510</td><td>0.2300</td><td>0.1120</td></tr><tr><td>Self Gen Title</td><td>0.4516</td><td>0.2602</td><td>0.3786</td><td>0.2174</td><td>0.3014</td><td>0.1844</td><td>0.1542</td><td>0.0616</td></tr><tr><td>std</td><td>0.0165</td><td>0.0179</td><td>0.0139</td><td>0.0117</td><td>0.0338</td><td>0.0272</td><td>0.0098</td><td>0.0130</td></tr><tr><td>majority</td><td>0.4420</td><td>0.2690</td><td>0.3720</td><td>0.2180</td><td>0.3230</td><td>0.2120</td><td>0.2600</td><td>0.2090</td></tr><tr><td>inclusion</td><td>0.4960</td><td>0.2950</td><td>0.3890</td><td>0.2380</td><td>0.3770</td><td>0.2140</td><td>0.2170</td><td>0.0950</td></tr><tr><td>Self Gen Title Desc</td><td>0.4682</td><td>0.2856</td><td>0.3714</td><td>0.2046</td><td>0.2814</td><td>0.1774</td><td>0.1878</td><td>0.0782</td></tr><tr><td>std</td><td>0.0085</td><td>0.0077</td><td>0.0086</td><td>0.0091</td><td>0.0261</td><td>0.0317</td><td>0.0322</td><td>0.0075</td></tr><tr><td>majority</td><td>0.4790</td><td>0.3070</td><td>0.3830</td><td>0.2050</td><td>0.3230</td><td>0.2370</td><td>0.2470</td><td>0.1720</td></tr><tr><td>inclusion</td><td>0.5020</td><td>0.3010</td><td>0.3810</td><td>0.2250</td><td>0.3610</td><td>0.1890</td><td>0.2430</td><td>0.1120</td></tr><tr><td>Self Gen Title Desc Guide</td><td>0.4536</td><td>0.2698</td><td>0.3854</td><td>0.1814</td><td>0.2652</td><td>0.0810</td><td>0.0890</td><td>0.0288</td></tr><tr><td>std</td><td>0.0248</td><td>0.0160</td><td>0.0078</td><td>0.0091</td><td>0.0194</td><td>0.0105</td><td>0.0179</td><td>0.0121</td></tr><tr><td>majority</td><td>0.4670</td><td>0.2940</td><td>0.3990</td><td>0.1860</td><td>0.3160</td><td>0.1450</td><td>0.1710</td><td>0.1550</td></tr><tr><td>inclusion</td><td>0.4850</td><td>0.2920</td><td>0.4060</td><td>0.2070</td><td>0.3140</td><td>0.1140</td><td>0.1820</td><td>0.0650</td></tr><tr><td>GPT Gen Title</td><td>0.4690</td><td>0.2654</td><td>0.3422</td><td>0.1804</td><td>0.2952</td><td>0.1818</td><td>0.1582</td><td>0.0654</td></tr><tr><td>std</td><td>0.0132</td><td>0.0138</td><td>0.0105</td><td>0.0089</td><td>0.0257</td><td>0.0261</td><td>0.0185</td><td>0.0121</td></tr><tr><td>majority</td><td>0.4690</td><td>0.2850</td><td>0.3250</td><td>0.1600</td><td>0.3020</td><td>0.1960</td><td>0.2070</td><td>0.1620</td></tr><tr><td>inclusion</td><td>0.5020</td><td>0.2830</td><td>0.3590</td><td>0.1960</td><td>0.3870</td><td>0.2260</td><td>0.2310</td><td>0.1060</td></tr><tr><td>GPT Gen</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Title Desc std</td><td>0.4574</td><td>0.2670</td><td>0.3716</td><td>0.2182</td><td>0.2834</td><td>0.1776</td><td>0.1758</td><td>0.0826</td></tr><tr><td>majority</td><td>0.0096</td><td>0.0131</td><td>0.0158</td><td>0.0144</td><td>0.0378</td><td>0.0296</td><td>0.0185</td><td>0.0115</td></tr><tr><td>inclusion</td><td>0.4740</td><td>0.2940</td><td>0.3940</td><td>0.2270</td><td>0.3330</td><td>0.2340</td><td>0.1950</td><td>0.1590</td></tr><tr><td>GPT Gen</td><td>0.4910</td><td>0.2760</td><td>0.3870</td><td>0.2250</td><td>0.3650</td><td>0.2090</td><td>0.2390</td><td>0.1190</td></tr><tr><td>Title Desc Guide</td><td>0.4500</td><td>0.2692</td><td>0.3804</td><td>0.1926</td><td>0.2804</td><td>0.1360</td><td>0.1236</td><td>0.0440</td></tr><tr><td>std</td><td>0.0198</td><td>0.0152</td><td>0.0138</td><td>0.0080</td><td>0.0299</td><td>0.0230</td><td>0.0133</td><td>0.0105</td></tr><tr><td>majority</td><td>0.4520</td><td>0.2810</td><td>0.3840</td><td>0.1820</td><td>0.2680</td><td>0.1560</td><td>0.2550</td><td>0.2160</td></tr><tr><td>inclusion</td><td>0.4650</td><td>0.2850</td><td>0.3830</td><td>0.2140</td><td>0.3610</td><td>0.1800</td><td>0.1850</td><td>0.0780</td></tr><tr><td>std</td><td>0.0132</td><td>0.0115</td><td>0.0050</td><td>0.0050</td><td>0.0186</td><td>0.0152</td><td>0.0410</td><td>0.0334</td></tr><tr><td>majority</td><td>0.4310</td><td>0.2300</td><td>0.2870</td><td>0.2870</td><td>0.2340</td><td>0.1860</td><td>0.2210</td><td>0.1910</td></tr><tr><td>inclusion</td><td>0.4640</td><td>0.2620</td><td>0.2930</td><td>0.2860</td><td>0.3430</td><td>0.1880</td><td>0.2320</td><td>0.1370</td></tr><tr><td>OrigDesc</td><td>0.3806</td><td>0.2112</td><td>0.2916</td><td>0.2844</td><td>0.2748</td><td>0.1872</td><td>0.1846</td><td>0.1186</td></tr><tr><td>std</td><td>0.0234</td><td>0.0174</td><td>0.0117</td><td>0.0125</td><td>0.0216</td><td>0.0173</td><td>0.0375</td><td>0.0372</td></tr><tr><td>majority</td><td>0.3820</td><td>0.2250</td><td>0.2770</td><td>0.2770</td><td>0.3440</td><td>0.3010</td><td>0.2770</td><td>0.2280</td></tr><tr><td>inclusion</td><td>0.4420</td><td>0.2480</td><td>0.3340</td><td>0.3070</td><td>0.3290</td><td>0.1850</td><td>0.2240</td><td>0.1210</td></tr><tr><td>Self Gen Title</td><td>0.3578</td><td>0.1880</td><td>0.2704</td><td>0.2434</td><td>0.1898</td><td>0.1254</td><td>0.1968</td><td>0.1612</td></tr><tr><td>std</td><td>0.0131</td><td>0.0078</td><td>0.0343</td><td>0.0314</td><td>0.0271</td><td>0.0172</td><td>0.0356</td><td>0.0338</td></tr><tr><td>majority</td><td>0.3390</td><td>0.1800</td><td>0.2750</td><td>0.2620</td><td>0.1570</td><td>0.1210</td><td>0.2970</td><td>0.2870</td></tr><tr><td>inclusion</td><td>0.4080</td><td>0.2350</td><td>0.3360</td><td>0.2610</td><td>0.2400</td><td>0.1340</td><td>0.2280</td><td>0.1430</td></tr><tr><td>Self Gen Title Desc</td><td>X</td><td>X</td><td>0.2542</td><td>0.2420</td><td>0.1898</td><td>0.1188</td><td>0.1724</td><td>0.1396</td></tr><tr><td>std</td><td>X</td><td>X</td><td>0.0166</td><td>0.0185</td><td>0.0284</td><td>0.0225</td><td>0.0321</td><td>0.0315</td></tr><tr><td>majority</td><td>X</td><td>X</td><td>0.2570</td><td>0.2570</td><td>0.2330</td><td>0.2050</td><td>0.2710</td><td>0.2570</td></tr><tr><td>inclusion</td><td>X</td><td>X</td><td>0.2900</td><td>0.2340</td><td>0.2410</td><td>0.1180</td><td>0.2160</td><td>0.1260</td></tr><tr><td>Self Gen Title Desc Guide</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.1444</td><td>0.1102</td></tr><tr><td>std</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.0248</td><td>0.0266</td></tr><tr><td>majority</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.2110</td><td>0.2080</td></tr><tr><td>inclusion</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.1980</td><td>0.1170</td></tr><tr><td>GPT Gen Title</td><td>0.3814</td><td>0.2058</td><td>0.2782</td><td>0.2544</td><td>0.1596</td><td>0.0998</td><td>0.1936</td><td>0.1488</td></tr><tr><td>std</td><td>0.0196</td><td>0.0270</td><td>0.0168</td><td>0.0114</td><td>0.0181</td><td>0.0203</td><td>0.0346</td><td>0.0306</td></tr><tr><td>majority</td><td>0.3680</td><td>0.2190</td><td>0.2770</td><td>0.2690</td><td>0.1450</td><td>0.1100</td><td>0.2720</td><td>0.2570</td></tr><tr><td>inclusion</td><td>0.4290</td><td>0.2440</td><td>0.3090</td><td>0.2340</td><td>0.2120</td><td>0.1120</td><td>0.2420</td><td>0.1430</td></tr><tr><td>GPT Gen</td><td>0.3720</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Title Desc</td><td></td><td>0.2172</td><td>0.2652</td><td>0.2544</td><td>0.1788</td><td>0.0958</td><td>0.1888</td><td>0.1394</td></tr><tr><td>std</td><td>0.0181</td><td>0.0136</td><td>0.0128</td><td>0.0180</td><td>0.0260</td><td>0.0249</td><td>0.0298</td><td>0.0200</td></tr><tr><td>majority</td><td>0.4040</td><td>0.2450</td><td>0.2760</td><td>0.2690</td><td>0.1650</td><td>0.1340</td><td>0.2800</td><td>0.2480</td></tr><tr><td>inclusion GPT Gen</td><td>0.4270</td><td>0.2460</td><td>0.3180</td><td>0.2720</td><td>0.2270</td><td>0.0930</td><td>0.2170</td><td>0.1250</td></tr><tr><td>Title Desc Guide</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.2008</td><td>0.1686</td></tr><tr><td>std</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.0343</td><td>0.0299</td></tr><tr><td>majority</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.2920</td><td>0.2870</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>inclusion</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.2350</td><td>0.1480</td></tr></table>

Table 2: SemEval – Three-Step classification results: macro F1 scores and std by model and prompting strategy.

Table 3: SemEval – Hybrid classification results: macro F1 scores and std by model and prompting strategy.

Table 4: SemEval – All-in-One classification results: macro F1 scores and std by model and prompting strategy.

<table><tr><td></td><td>All</td><td>All</td><td>No Granite</td><td>No Granite</td></tr><tr><td colspan="2"></td><td>F1 es F1 en</td><td>F1 es</td><td>F1 en</td></tr><tr><td colspan="2">Dipro-Title maj.</td><td>0.6428 0.5977</td><td>0.6414</td><td>0.5869</td></tr><tr><td colspan="2">inc.</td><td>0.4001 0.3937</td><td>0.5176</td><td>0.4662</td></tr><tr><td colspan="2">Dipro-OrigDesc maj.</td><td>0.6341 0.6046</td><td>0.6417</td><td>0.6216</td></tr><tr><td colspan="2">inc.</td><td>0.4048 0.3975</td><td>0.5375</td><td>0.4960</td></tr><tr><td colspan="2">SemEval-Title maj.</td><td>F1 nar F1 subnar 0.2870 0.2870</td><td>F1 nar 0.3000</td><td>F1 subnar</td></tr><tr><td colspan="2">(all_in_one) inc.</td><td>0.3280</td><td>0.1930 0.3910</td><td>0.3000 0.2410</td></tr><tr><td colspan="2">SemEval-Title maj.</td><td>0.2900</td><td>0.2890 0.3120</td><td>0.2970</td></tr><tr><td colspan="2">(hybrid) inc.</td><td>0.3270 0.1770</td><td>0.3800</td><td>0.2310</td></tr><tr><td colspan="2">SemEval-Title maj.</td><td>0.4890 0.3780</td><td>0.4470</td><td>0.2930</td></tr><tr><td colspan="2">(3_steps) inc.</td><td>0.3100</td><td>0.1990 0.3960</td><td>0.2610</td></tr><tr><td colspan="2">SemEval-OrigDesc maj.</td><td>0.2990</td><td>0.2960 0.3280</td><td>0.3260</td></tr><tr><td colspan="2">(all_in_one) inc.</td><td>0.3200</td><td>0.1840 0.3720</td><td>0.2320</td></tr><tr><td colspan="2">SemEval-OrigDesc maj.</td><td>0.2280</td><td>0.2280 0.1490</td><td>0.1490</td></tr><tr><td colspan="2">(hybrid) inc.</td><td>0.2640</td><td>0.1380 0.2810</td><td>0.1610</td></tr><tr><td colspan="2">SemEval-OrigDesc maj.</td><td>0.5090</td><td>0.4000</td><td>0.5560 0.3900</td></tr><tr><td colspan="2">(3_steps) inc.</td><td>0.3030</td><td>0.1960</td><td>0.4170 0.2800</td></tr></table>

Table 5: Ensemble performance for the Title and OriginalDesc prompting strategies, using all classifier models (All) and excluding Granite (No Granite).

In Dipromats, employing original human-authored descriptions (OrigDesc) leads to an average performance improvement of 1.5 pp (percentage points) over using only narrative titles (Title). In contrast, using generated descriptions results in a mean decrease of approximately 4 pp. A more granular analysis reveals that the Self\_Gen\_Title and GPT\_Gen\_Title strategies yield a 3.4 pp drop, Self\_Gen\_Title\_Desc and GPT\_Gen\_Title\_Desc yield a 0.7 pp drop, and Self\_Gen\_Title\_Desc\_Guide and GPT\_Gen\_Title\_Desc\_Guide result in more substantial degradations of 6 pp and 10.6 pp, respectively.

SemEval exhibits a similar trend. Using original descriptions leads to a modest 0.66 pp improvement over using titles alone. However, generated descriptions underperform by an average of 2.7 pp. Specifically, Self\_Gen\_Title, GPT\_Gen\_Title, Self\_Gen\_Title\_Desc, and GPT\_Gen\_Title\_Desc each produce an average decline of 2.6 pp, with negligible diferences between them. The more detailed prompt variants, Self\_Gen\_Title\_Desc\_Guide and GPT\_Gen\_Title\_Desc\_Guide, result in 4.8 pp reductions, excluding the All-in-One setting which only the smallest model could process.

Overall, 28.44% of generated approaches outperform title-only prompts, while 71.56% do not. Comparing only Title vs. OrigDesc, the latter proves superior in 56% of cases. When contrasting OrigDesc with generated descriptions, OrigDesc performs better in 87.1% of configurations.

Conclusion: While titles alone ofer a reasonable baseline, original human authored descriptions consistently enhance performance. Automatically generated content, particularly when verbose, may introduce noise that ofsets potential contextual gains.

RQ2: Does the use of original human-created narrative descriptions produce better results than automatically rewritten or improved versions generated by the models themselves?

Building on the approach used in RQ1, this analysis focuses on the diference in performance between each generated prompt configuration and the OrigDesc baseline. By computing and averaging these deltas across the same model and dataset settings, we assess whether automatically rewritten descriptions ofer any consistent advantage over original human-authored texts.

As discussed above, OrigDesc yields the highest performance overall. In Dipromats, prompt strategies involving generated descriptions result in a 5.6 pp average decline. When using full descriptions (Title\_Desc and Title\_Desc\_Guide), performance drops by 7.5 pp and 12 pp, respectively. In SemEval, the decrease is 4.1 pp for narrative detection and 2.7 pp for subnarrative detection.

Only 12.9% of all evaluated prompt configurations outperform the OrigDesc baseline.

Additionally, in deployment-faithful scenarios where only titles or noisy/mismatched descriptions are available, model performance is expected to decrease. Our experiments show that manually curated narrative descriptions consistently outperform automatically generated prompts. Introducing automatically generated noisy descriptions would not alter this observation, but systematically evaluating their impact falls outside the scope of this work.

Conclusion: Automatically rewritten prompts, even when designed to be more informative, rarely surpass the efectiveness of original human-curated narrative descriptions.

RQ2.1: Among the strategies for prompt improvement, which yields the highest performance in narrative detection tasks?

As in RQ1, we compute performance deltas between each generated prompt strategy and the Title baseline. These diferences are then averaged across all model–dataset–task configurations to identify which strategies ofer the most consistent improvements. For example, for the Self\_Gen\_Title strategy, we calculate the average of 32 F1-score diferences, corresponding to 4 models × 8 evaluation points (2 languages in Dipromats and 3 classification schemes × 2 tasks in SemEval).

When comparing performance diferences relative to Title-only prompts, Self\_Gen\_Title\_Desc and GPT\_Gen\_Title\_Desc are the best-performing generated strategies, with average deltas of 1.89 pp and 1.87 pp, respectively. These approaches leverage both the title and the original description to construct a richer, yet controlled, narrative context.

This performance gain is intuitive: the improved prompts preserve the editorial framing of the original while refining it. Conversely, the Title\_Desc\_Guide variants—though semantically deeper—may introduce extraneous information that dilutes the core narrative signal.

Conclusion: The most efective prompt improvements are those that augment, rather than radically reformulate, the original narrative context.

RQ3: Does the narrative detection performance improve when using self-generated narrative descriptions, or is it preferable to rely on descriptions produced by more capable models, such as GPT systems?

To address this research question, we directly compare prompt variants that difer only in the model used to generate the narrative descriptions, either self-generated by the classifier model itself or produced by a GPT-based system. Specifically, we calculate the F1-score diferences between each pair of matching strategies, such as Self\_Gen\_Title vs. GPT\_Gen\_Title, across all model and task configurations. These deltas are then averaged to assess whether one generation source consistently outperforms the other.

In Dipromats, no consistent performance advantage is observed between selfgenerated and GPT-generated descriptions across the Gen\_Title and Gen\_Title\_Desc strategies. However, for Gen\_Title\_Desc\_Guide, performance is similar across models, except in the case of Gemma, where self-generated prompts outperform GPT ones by 18 pp.

In SemEval, most diferences are negligible. Two exceptions stand out: in the three-step classification setting, Self\_Gen\_Title exceeds GPT\_Gen\_Title by 5 pp, and Self\_Gen\_Title\_Desc\_Guide surpasses GPT\_Gen\_Title\_Desc\_Guide by 2.6 pp.

Conclusion: On average, there is no significant diference, although there may be slight benefits to using self-generated descriptions.

RQ4: How efective are ensemble strategies in improving the robustness and accuracy of zero-shot narrative detection?

Across individual model ensembles (i.e., same model–prompt combinations across 5 runs), SemEval shows an average improvement of 2.9 pp. In Dipromats, majoritybased ensembles ofer a marginal 0.6 pp gain, while inclusion-based ensembles perform 0.6 pp worse.

In the cross-model ensemble setting, we evaluated combinations using all models (All) and excluding Granite (No\_Granite). Only the Title and OrigDesc prompt strategies were considered, given their superior performance and comparable results. To evaluate the efectiveness of ensemble strategies, we compared the average performance of model specific ensembles, created by aggregating five runs per model, with a cross-model ensemble combining predictions from all runs across models (15 for No\_Granite, 20 for All).

In Dipromats, majority based ensembles consistently outperform the mean of model specific ensembles, often exceeding even the best individual run. In contrast, inclusion based ensembles underperform in all configurations, likely due to an increased rate of false positives when any of the 15 or 20 runs erroneously predicts an extra label.

In SemEval, majority based ensembles outperform the baseline in 62.5% of cases, while inclusion ensembles improve results in only 20.8%. Notably, inclusion ensembles degrade performance by up to 3.5 pp, particularly in subnarrative detection. Majority voting proves beneficial in this task, boosting performance by 8.5 pp with four models and 4.8 pp with three. However, even the best ensemble occasionally underperforms relative to top scoring individual runs, possibly due to the large output space and inherent dificulty of the task (evidenced by low interannotator agreement).

Conclusion: Ensemble methods—particularly those based on majority voting substantially enhance robustness and accuracy, especially for fine-grained subnarrative detection, and can rival or surpass State Of The Art models trained with supervision.

RQ5: How does the diference in model size afect performance in narrative identification?

To assess model stability across prompt variations, we used the performance deltas computed in RQ1, i.e., the macro F1 diferences between each prompt strategy and the Title strategy. By averaging these deltas separately for each classification model, we were able to estimate how sensitive each model is to prompt changes. This allows us to quantify performance volatility in relation to model size and architecture.

The Calme 78B model demonstrates the smallest performance variance across prompt strategies, with an average degradation of under 1 pp. In contrast, smaller models such as Gemma, Exaone, and Granite exhibit greater instability, with an average drop of 3.7 pp. In terms of ensemble stability, Gemma 12B is the most consistent across both datasets, followed by Calme 78B. Exaone and Granite show the greatest fluctuations.

Across all experiments, the standard deviation analysis reveals clear stability patterns among models and prompting strategies. Gemma and Calme consistently demonstrate the lowest variability, indicating robust and stable performance across tasks, while Exaone and especially Granite exhibit greater sensitivity to prompt structure and language context.

Conclusion: Larger models are generally more robust to prompt variation, but some mid-sized models strike an efective balance between stability and computational cost.

RQ6: Are the narrative descriptions generated by LLMs suficiently coherent, semantically accurate, and pragmatically acceptable to serve as reliable inputs for narrative detection?

A qualitative inspection suggests that most generated descriptions are semantically plausible and syntactically well-formed. However, prompts relying solely on titles may yield descriptions that deviate subtly from the annotators’ intended framing, potentially altering narrative polarity or scope.

# Example of Semantic Drift in Generated Descriptions

## Generated Description EU6 (Granite, Using only Title):

This narrative explores the European Union’s (EU) positive impact on global affairs throughout 2020, focusing on its contributions to social, political, and cultural advancements. The EU navigates the challenges presented by the COVID-19 pandemic while maintaining its commitment to solidarity and cooperation, showcasing its role as a significant global player. The narrative likely delves into EU initiatives aimed at promoting healthcare equity, fostering economic resilience, safeguarding human rights, and preserving cultural heritage amidst the turmoil. Additionally, it may address the EU’s eforts to mediate international conflicts and support sustainable development, emphasizing its multilateral approach to addressing global issues. This account underscores the EU’s influence as a collective entity in shaping a more harmonious and equitable world, ofering insights into its diverse member states’ collaboration and unity during a tumultuous year.

## Original Human-Authored Descriptions:

• EU3: Classify tweets that promote EU policies and their benefits, demonstrating how European measures tangibly improve the lives of its citizens.

• EU4: Detect tweets that position the EU as a leader in tackling contemporary and future challenges such as climate change, digitalisation, and regulation.

• EU6: Find tweets that illustrate the EU’s contributions to global development, peace, safety, and well-being, emphasizing its role as a key partner for neighbouring and global societies.

As shown, the generated text tends to blend multiple themes from related narratives and expand them beyond the scope of the original EU6 title. While this output remains plausible and linguistically fluent, it risks obscuring the intended boundaries of the target narrative, especially in multi-label or fine-grained classification settings.

Conclusion: LLM-generated narrative descriptions can serve as reliable inputs, but their utility depends on alignment with the annotation framework. Overly abstract or generalised generations may diverge from ground-truth labelling logic.

## 5.2. Confidence Intervals

Quantifying confidence intervals is essential for validating claims regarding the robustness of the models. Performing formal statistical tests or deriving confidence intervals for aggregated scores would typically require a large number of independent runs, which is computationally impractical. An alternative approach, inspired by [31], allows for approximate confidence interval estimation using a resampling methodology. This method leverages the gold-standard labels together with the model predictions for each instance to perform bootstrap sampling with replacement, calculating a score for each resampled set. The resulting distribution of scores is then used to determine the 95% confidence interval via the percentile method.

As the gold-standard labels for SemEval are not publicly available, this analysis has been conducted on the Dipromats dataset. For each model and prompting strategy, the predictions from the five independent runs were concatenated, and 2000 bootstrap samples of 800 instances (equal to the test set size) were generated. The resulting 95% confidence intervals for each configuration are presented in Table 6.

<table><tr><td></td><td>Lang</td><td>Calme</td><td>Gemma</td><td>Exaone</td><td>Granite</td></tr><tr><td rowspan="2">Title</td><td>es</td><td>(0.612, 0.657)</td><td>(0.591, 0.634)</td><td>(0.528, 0.575)</td><td>(0.306, 0.359)</td></tr><tr><td>en</td><td>(0.560, 0.606)</td><td>(0.544, 0.585)</td><td>(0.474, 0.522)</td><td>(0.345, 0.391)</td></tr><tr><td rowspan="2">OrigDesc</td><td>es</td><td>(0.615, 0.662)</td><td>(0.601, 0.648)</td><td>(0.548, 0.597)</td><td>(0.310, 0.359)</td></tr><tr><td>en</td><td>(0.591, 0.637)</td><td>(0.572, 0.613)</td><td>(0.512, 0.557)</td><td>(0.328, 0.375)</td></tr><tr><td rowspan="2">Self_Gen_Title</td><td>es</td><td>(0.599, 0.643)</td><td>(0.555, 0.598)</td><td>(0.483, 0.530)</td><td>(0.264, 0.309)</td></tr><tr><td>en</td><td>(0.548, 0.593)</td><td>(0.511, 0.552)</td><td>(0.462, 0.505)</td><td>(0.276, 0.320)</td></tr><tr><td rowspan="2">Self Gen Title Desc</td><td>es</td><td>(0.608, 0.656)</td><td>(0.564, 0.608)</td><td>(0.522, 0.569)</td><td>(0.266, 0.311)</td></tr><tr><td>en</td><td>(0.596, 0.641)</td><td>(0.549, 0.591)</td><td>(0.488, 0.537)</td><td>(0.306, 0.352)</td></tr><tr><td rowspan="2">Self Gen Title Desc Guide</td><td>es</td><td>(0.607, 0.656)</td><td>(0.512, 0.557)</td><td>(0.432, 0.483)</td><td>(0.221, 0.270)</td></tr><tr><td>en</td><td>(0.583, 0.629)</td><td>(0.455, 0.500)</td><td>(0.402, 0.451)</td><td>(0.260, 0.306)</td></tr><tr><td rowspan="2">GPT Gen Title</td><td>es</td><td>(0.586, 0.632)</td><td>(0.563, 0.606)</td><td>(0.502, 0.548)</td><td>(0.245, 0.290)</td></tr><tr><td>en</td><td>(0.542, 0.589)</td><td>(0.502, 0.545)</td><td>(0.470, 0.512)</td><td>(0.275, 0.320)</td></tr><tr><td rowspan="2">GPT Gen Title Desc</td><td>es</td><td>(0.618, 0.663)</td><td>(0.592, 0.637)</td><td>(0.524, 0.568)</td><td>(0.261, 0.309)</td></tr><tr><td>en</td><td>(0.592, 0.637)</td><td>(0.559, 0.599)</td><td>(0.503, 0.546)</td><td>(0.276, 0.324)</td></tr><tr><td rowspan="2">GPT Gen Title Desc Guide</td><td>es</td><td>(0.603, 0.651)</td><td>(0.240, 0.310)</td><td>(0.460, 0.509)</td><td>(0.198, 0.243)</td></tr><tr><td>en</td><td>(0.582, 0.627)</td><td>(0.347, 0.412)</td><td>(0.414, 0.461)</td><td>(0.249, 0.294)</td></tr></table>

Table 6: 95% confidence intervals for all model and prompting configurations on the Dipromats test set

Across models, languages, and prompting strategies, the 95% confidence intervals exhibit broadly comparable widths, indicating a similar degree of variability in performance. To identify the strongest configurations, it is therefore most appropriate to focus on the interval midpoints. Notably, the configurations with the highest midpoints correspond exactly to those that achieve the best aggregated mean F1 scores in Table 1, confirming the stability of the rankings obtained from averaging the five runs. This alignment suggests that the aggregated means provide a reliable basis for comparative claims.

![](images/9653202763517e66e3f02965539a994048c07632c0fc4e12acb698ce577338bc.jpg)  
Figure 2: A tweet for which only 1 out of 160 predictions across all models and prompt strategies is correct.

## 5.3. Qualitative Error Analysis

To further interpret the quantitative findings, a qualitative error analysis was conducted to examine specific cases where the models succeed or fail in identifying narratives. This analysis is only feasible for the Dipromats dataset, as it is the only collection for which gold-standard labels and the original English tweets are available. Only tweets containing at least one yes label for any narrative were considered. The goal is to explore the types of errors that arise under diferent prompting strategies and model configurations, as well as to understand how narrative descriptions influence classification outcomes.

## 5.3.1. Cross-model and cross-prompt variability

The analysis jointly considers predictions from the four evaluated models, each tested with eight prompting strategies and five independent runs, yielding a total of 160 predictions per tweet. Tweets such as the one shown in Figure 2 were correctly classified in only one out of 160 predictions. This illustrates the high degree of subjectivity in some tweets, where the stance toward a narrative is subtle, implicit, or context-dependent. Overall, 26 out of the 562 instances were correctly identified in fewer than 10% of all predictions, reflecting intrinsic ambiguity.

![](images/beaf018325c85cce80a6b0bf0f542284b29ee834dfcf378e9839847bece4d439.jpg)

Figure 3: A tweet correctly classified in all runs by the largest model (Calme) but showing sub stantially lower accuracy for smaller models.  
![](images/0caafe6d19012be067c918ad4de7546d1a321775df647894b13da7a70961b3d1.jpg)  
Figure 4: A tweet that the largest model fails to classify correctly, whereas smaller models achieve partial success.

In contrast, tweets such as the one illustrated in Figure 3 achieved 100% accuracy across all runs performed with the Calme model, yet displayed markedly lower performance for the remaining models. In the opposite case, tweets like that in Figure 4 show the reverse pattern: the largest model fails across all runs, while smaller models achieve correct predictions. However, even in these cases, the accuracy for the smaller models rarely exceeds 30%.

## 5.3.2. Focusing on the Calme model: the role of narrative descriptions.

A deeper inspection was carried out for the Calme model to evaluate the influence of narrative prompting. Specifically, performance diferences between using only the narrative title (Title) and other prompting strategies involving extended descriptions (original or automatically generated) were examined.

Two representative examples are presented in Figure 5, which are the only tweets in the dataset for which all five runs using only the Title prompt correctly identify the narratives, while none of the remaining 35 runs (those employing full or generated descriptions) produce the correct label. Analogously, Figure 6 displays the only two tweets that exhibit the opposite behaviour—all 35 runs using narrative descriptions achieve correct predictions, whereas none of the five Title-only runs succeed.

![](images/86750e1c9e2904e844739f81feb2a10256a73e99a4cc4945d77ce8f80ce3dd99.jpg)  
Figure 5: Two tweets correctly classified in all Title-only runs but misclassified in all other runs using original or generated descriptions.

To summarise these contrasts numerically, Table 7 reports the number of instances meeting each of the following conditions:

• C1: All Title runs are correct and all compared-prompt runs fail.

• C2: All Title runs fail and all compared-prompt runs are correct.

• C3: Some Title runs succeed while all compared-prompt runs fail.

• C4: All Title runs fail while some compared-prompt runs succeed.

The numerical contrasts in Table 7 indicate that incorporating narrative descriptions—whether original or automatically generated—can substantially alter the predictive behaviour compared to using the title alone. For instance, richer prompt strategies such as Self\_Gen\_Title\_Desc\_Guide and GPT\_Gen\_Title\_Desc\_Guide exhibit higher counts in C2 and C4, reflecting cases where these descriptions enable correct predictions that the title-only runs fail to capture. Conversely, elevated values in C1 suggest that in some instances the additional descriptions may actually reduce performance relative to the concise title, potentially due to noise or over-specification introduced by the model-generated content.

![](images/8c4aea9a9b80b417229e3b1790d21821e7a017e001279741d77ec94e2f6aac4e.jpg)  
Figure 6: Two tweets correctly classified in all runs using original or generated descriptions but misclassified in all Title-only runs.

<table><tr><td></td><td>C1</td><td>C2</td><td>C3</td><td>C4</td></tr><tr><td rowspan="7">OrigDesc Self Gen Title Self Gen Title Desc Self Gen Title Desc</td><td>11</td><td>20</td><td>39</td><td>35</td></tr><tr><td>15</td><td>12</td><td>46</td><td>30</td></tr><tr><td>20</td><td>27</td><td>43</td><td>44</td></tr><tr><td>Guide 28</td><td>22</td><td>60</td><td>37</td></tr><tr><td>16</td><td>14</td><td>50</td><td>33</td></tr><tr><td>13</td><td>22</td><td>39</td><td>37</td></tr><tr><td>Guide 27</td><td>33</td><td>56</td><td>49</td></tr></table>

Table 7: Comparison between Title-only and other prompting strategies for the Calme model.

## 6. Comparison with the State of the Art

To provide a head-to-head comparison with state-of-the-art systems, Tables 8 and 9 report the performance of our best zero-shot model alongside the top leaderboard submissions from SemEval and Dipromats, respectively. In both tables, $\Delta$ denotes the diference in performance relative to the best-performing system.

<table><tr><td>System</td><td>Setting</td><td>F1 nar</td><td>F1 subnar</td><td>∆ F1 nar</td><td>∆ F1 subnar</td></tr><tr><td>GATENLP COGNAC</td><td>Supervised Zero-Shot</td><td>0.6270</td><td>0.4630</td><td></td><td></td></tr><tr><td></td><td></td><td>0.5540</td><td>0.4260</td><td>-0.073</td><td>-0.037</td></tr><tr><td>INSALyon2</td><td>Zero-Shot</td><td>0.5130</td><td>0.4060</td><td>-0.114</td><td>-0.057</td></tr><tr><td>Our System iLostTheCode</td><td>Zero-Shot</td><td>0.5090</td><td>0.4000</td><td>-0.118</td><td>-0.063</td></tr><tr><td></td><td>Supervised</td><td>0.4980</td><td>0.3730</td><td>-0.129</td><td>-0.090</td></tr><tr><td>KostasThesis</td><td>Supervised</td><td>0.5560</td><td>0.3620</td><td>-0.071</td><td>-0.101</td></tr><tr><td>NCLteam</td><td>Supervised</td><td>0.4860</td><td>0.3450</td><td>-0.141</td><td>-0.118</td></tr><tr><td>Narrlangen</td><td>Zero-Shot</td><td>0.4440</td><td>0.3440</td><td>-0.183</td><td>-0.119</td></tr><tr><td>PATeam</td><td>Supervised</td><td>0.5210</td><td>0.3390</td><td>-0.106</td><td>-0.124</td></tr><tr><td>PingAnAI</td><td>Supervised</td><td>0.5210</td><td>0.3390</td><td>-0.106</td><td>-0.124</td></tr></table>

Table 8: Head-to-head comparison of our best zero-shot system against leaderboard systems on SemEval 2025 Task 10 (Subtask 2).

<table><tr><td>System</td><td>Setting</td><td>F1 es</td><td>F1 en  $\Delta \mathbf { F } \mathbf { 1 }$ </td><td>es  $\Delta \mathbf { F } \mathbf { 1 }$  en</td></tr><tr><td>OB1</td><td>Few-Shot Few-Shot</td><td>0.6854 0.6847</td><td>0.6701 -0.0007</td><td>-0.0334</td></tr><tr><td>UNED-JF Our System ManchaAzul UNED-AP</td><td>Zero-Shot Few-Shot Multi-Agent Linear Transformation Few-Shot</td><td>0.6453 0.6403 0.4993</td><td>0.6367 0.6209 0.6117 0.5226 -0.1861</td><td>-0.0401 -0.0492 -0.0451 -0.0584 -0.1475</td></tr></table>

Table 9: Head-to-head comparison of our best zero-shot system against leaderboard systems on Dipromats 2024 Task 2.

For SemEval, participating systems covered both supervised and zero-shot strategies. The GATENLP team [20] achieved the highest overall performance by implementing a hierarchical three-step prompting framework combined with fine-tuning of a LLaMa 3.2 model [32]. Their results demonstrate the clear advantage of utilising task-specific training data, as their zero-shot experiments yielded substantially lower scores. The COGNAC [32] system adopted a diferent strategy, first summarising news articles into texts of 300 words or fewer, while explicitly retaining key topics, sentiments, and narratives. Narrative detection was then formulated as a binary classification task, followed by subnarrative detection for those narratives identified as supported. Both stages were implemented in a zero-shot setting using GPT-4o-mini and LLaMa 3.1-8B-Instruct. However, their results are not fully comparable to other submissions, as no explicit topic classification module was reported; it can be inferred that topic identification was carried out using the original filenames of the news files. In our own experiments, topic classification accuracy on the training and development sets ranged between 0.8 and 0.9, highlighting the potential impact of this assumption on overall performance. The INSALyon2 [22] team also adopted a zero-shot methodology, employing a multi-agent framework based on GPT-4o and GPT-4o-mini. By contrast, the remaining participants relied on supervised approaches, training their systems with the task-specific data provided. Within this context, our proposed system demonstrates competitive performance without the need for any training data, reinforcing its efectiveness as a zero-shot solution.

In the Dipromats leaderboard, the OB1 team reports the highest score, although no details are provided regarding the underlying model or methodological choices, which limits comparability. The UNED-JF team employed the calme-2.4-rys-78b model quantized to 8 bits and adopted a few-shot strategy using only the titles of the narratives. The ManchaAzul team [33] adopted a multi-agent few-shot strategy with GPT-4, while UNED-AP transformed the vector space of tweet embeddings to perform classification. The oficial baseline relied on a simple zero-shot approach using Mixtral 8x7B [34], and the UMU-Team [35] implemented a few-shot strategy with Zephyr-7B-beta [36]. Within this competitive landscape, our system achieves robust performance, surpassing models based on GPT-4 despite its comparatively lower computational requirements.

## 7. Generalisation of Results to Other Languages

Although our main experiments are conducted in a limited set of languages, it is important to assess whether the proposed unsupervised narrative detection framework generalises beyond this setting. Leveraging the multilingual capabilities of large generative models, we examine whether our previous conclusions hold when analysing texts in other languages. To this end, we consider two complementary settings: providing the models with documents in their original language, and supplying translated versions of the same texts as input.

## 7.1. Experimental setting

To investigate the generalisation of our approach to other languages, we use the development set of the SemEval dataset, which includes documents written in Bulgarian, Portuguese, Hindi, and Russian. The distribution of instances per language is as follows: 35 documents each in Bulgarian, Portuguese, and Hindi, and 32 in Russian.

For this exploratory evaluation, we employ the Calme 78B model and the hybrid classification approach, as it achieved the strongest performance in the experiments presented in Section 5. We explore two alternative input strategies. In the first setting, the model is provided with the news article in its original language, allowing us to evaluate whether it can directly leverage its multilingual representations to identify narratives without any language normalisation. In the second setting, the same articles are automatically translated into English using OPUS models [37] and then used as input to the model. This second approach allows us to study transfer to languages that the model has not been exposed to during training.

Given that the aim of this analysis is to provide a lightweight yet informative multilingual slice rather than a full-scale replication of the pipeline, we run a single inference pass per document instead of the five-run setting used elsewhere in the paper.

## 7.2. Feeding models with input texts in other languages

Table 10 summarises the model’s performance when using the documents in their original languages (Bulgarian, Portuguese, Hindi, and Russian). For each configuration, the first column reports the F1 score for narrative prediction, while the second column reports the F1 score for subnarrative detection.

## 7.3. Feeding models with translations into English

Table 11 presents the results when the documents are translated into English. The same reporting scheme applies: the first column indicates the F1 score for narrative detection, and the second column the F1 score for subnarrative identification.

## 7.4. Results

Across both original and translated documents, a clear pattern emerges: the most efective prompting strategies are consistently those that use only the narrative title or combine the title with the original human-authored description. Automatically generated descriptions do not appear to ofer a meaningful improvement over these simpler strategies. This suggests that the large language models possess suficient contextual and pragmatic knowledge to identify narratives without relying on augmented or automatically generated narrative descriptions.

<table><tr><td></td><td>BG</td><td>PT</td><td></td><td>HI</td><td>RU</td></tr><tr><td>Title</td><td>0.5276 0.3659</td><td>0.4314 0.2057</td><td>0.4007</td><td>0.2244</td><td>0.5375 0.3382</td></tr><tr><td>OrigDesc</td><td>0.5124 0.3186</td><td>0.5295 0.2129</td><td>0.4101</td><td>0.2528</td><td>0.5983 0.3383</td></tr><tr><td>Self Gen Title</td><td>0.5571 0.3477</td><td>0.3805 0.1129</td><td>0.3720</td><td>0.1940</td><td>0.5012 0.2713</td></tr><tr><td>Self Gen Title Desc</td><td>0.6076 0.3524</td><td>0.3948 0.1333</td><td>0.4222</td><td>0.1962</td><td>0.5321 0.2494</td></tr><tr><td>Self Gen Title Desc Guide</td><td>0.5629 0.3015</td><td>0.3981 0.1610</td><td>0.3981</td><td>0.2467</td><td>0.5316 0.2657</td></tr><tr><td>GPT Gen Title</td><td>0.5000 0.2877</td><td>0.4133</td><td>0.1571 0.3773</td><td>0.1994</td><td>0.5114 0.2540</td></tr><tr><td>GPT Gen Title Desc</td><td>0.5600 0.3467</td><td>0.4097</td><td>0.1034 0.3392</td><td>0.2112</td><td>0.5005 0.2735</td></tr><tr><td>GPT Gen Title Desc Guide</td><td>0.4924 0.2273</td><td>0.3649</td><td>0.1533 0.3554</td><td>0.1673</td><td>0.5158 0.3113</td></tr></table>

Table 10: Evaluation results using the original language documents on the SemEval development set.

These findings reinforce the earlier conclusions drawn from the English and Spanish experiments, indicating that human-curated descriptions provide a robust signal, whereas additional automatically generated content does not systematically enhance zero-shot classification performance.

## 8. Scalability Analysis

To assess the practical feasibility of the proposed zero-shot narrative detection methodology, a limited scalability analysis was conducted. While the main focus of this study is on evaluating the intrinsic capabilities of LLMs in narrative identification, it is important to characterise the computational requirements and eficiency of diferent models and prompt strategies.

Tables 12 and 13 summarise the computational footprint for the OrigDesc prompt strategy across the four models. For each dataset, the number of requests sent to the model, the average time per request, and the total GPU time are reported. All experiments were executed on a high-end workstation with 4 × NVIDIA RTX A5000 GPUs (24 GB GDDR6 each) and 256 GB of system RAM.

To illustrate the variability in input length, Figure 7 shows the distribution of the number of tokens per document for the OrigDesc prompt strategy on SemEval using All-in-One approach, across all four models. While Dipromats tweets are short and exhibit limited variability in token counts, SemEval news articles display a wider range, which afects both memory usage and inference latency.

<table><tr><td></td><td>BG</td><td>PT</td><td>HI</td><td></td><td>RU</td></tr><tr><td>Title</td><td>0.6078 0.4248</td><td>0.4771 0.2257</td><td>0.3329</td><td>0.2305</td><td>0.6691 0.4312</td></tr><tr><td>OrigDesc</td><td>0.5695 0.3356</td><td>0.4062 0.1635</td><td>0.4016</td><td>0.2939</td><td>0.4432 0.3057</td></tr><tr><td>Self Gen Title</td><td>0.5659 0.3729</td><td>0.4124 0.1659</td><td>0.3473</td><td>0.2080</td><td>0.5050 0.2712</td></tr><tr><td>Self Gen Title Desc</td><td>0.5263 0.2997</td><td>0.3301 0.1502</td><td>0.3109</td><td>0.1852</td><td>0.5308 0.2720</td></tr><tr><td>Self Gen Title Desc Guide</td><td>0.5524 0.3003</td><td>0.3962 0.1834</td><td>0.3724</td><td>0.2060</td><td>0.5222 0.2937</td></tr><tr><td>GPT Gen Title</td><td>0.5452 0.3702</td><td>0.4350 0.1610</td><td>0.3074</td><td>0.1727</td><td>0.4915 0.2399</td></tr><tr><td>GPT Gen Title Desc</td><td>0.5563 0.3710</td><td>0.4873 0.1529</td><td>0.3750</td><td>0.2166</td><td>0.5292 0.2135</td></tr><tr><td>GPT Gen Title Desc Guide</td><td>0.5743 0.3210</td><td>0.3135 0.1105</td><td>0.2718</td><td>0.1532</td><td>0.4820 0.2334</td></tr></table>

Table 11: Evaluation results using English translations of the SemEval development set documents.
<table><tr><td>Model</td><td>Avg  $\mathbf { T i m e / R e q }$   $\# \mathrm { R e q }$  (s)</td><td>Total GPU Time (s)</td></tr><tr><td>Calme</td><td>1600 18</td><td>28800</td></tr><tr><td>Gemma</td><td>1600 8</td><td>12800</td></tr><tr><td>Exaone</td><td>1600 5</td><td>8000</td></tr><tr><td>Granite</td><td>1600 3</td><td>4800</td></tr></table>

Table 12: Summary of the OrigDesc processing statistics for the Dipromats dataset.

As shown in Figure 7, it can be observed that, as model size decreases, the number of tokens that needs to be processed increases, reflecting diferences in how models encode and interpret the narrative context.

## 9. Conclusion

This study explores the ability of large pre-trained linguistic models to interpret and classify the underlying narratives in social messages without the need for task-specific training. Our zero-shot experiments across the Dipromats and SemEval datasets confirm that the provision of human-written narrative descriptions, consistently enhances performance compared to title-only prompts. While automatically generated descriptions are grammatically fluent and often semantically rich, they frequently introduce subtle shifts in framing that can degrade classification accuracy, especially in multilabel and fine-grained subnarrative scenarios.

<table><tr><td>Model</td><td>Classification Scheme</td><td>Avg Time/Req # Req (s)</td><td>Total GPU Time (s)</td></tr><tr><td rowspan="3">Calme</td><td>All-in-One 101</td><td>61</td><td>6180</td></tr><tr><td>Hybrid 202</td><td>52</td><td>10504</td></tr><tr><td>3-Step 417</td><td>44</td><td>18348</td></tr><tr><td rowspan="3">Gemma</td><td>All-in-One 101</td><td>23</td><td>2323</td></tr><tr><td>Hybrid 202</td><td>17</td><td>3434</td></tr><tr><td>3-Step 543</td><td>14</td><td>7602</td></tr><tr><td rowspan="3">Exaone</td><td>All-in-One 101</td><td>18</td><td>1818</td></tr><tr><td>Hybrid 202</td><td>15</td><td>3030</td></tr><tr><td>3-Step 401 101</td><td>9</td><td>3609</td></tr><tr><td rowspan="3">Granite</td><td>All-in-One</td><td>7</td><td>707</td></tr><tr><td>Hybrid 202 482</td><td>5</td><td>1010</td></tr><tr><td>3-Step</td><td>3</td><td>1446</td></tr></table>

Table 13: Summary of the OrigDesc processing statistics for the SemEval dataset.

Ensemble strategies, and in particular majority-voting across diverse model runs, further bolster robustness and frequently surpass the best individual model performances. The greatest gains appear in subnarrative detection, where aggregation reduces the impact of model-specific variability. Importantly, excluding the smallest model from the ensemble yields a more stable outcome, indicating a need to balance diversity with baseline reliability.

Model scale emerges as a key factor: the largest architectures not only exhibit the least sensitivity to prompt variation, but also achieve the highest overall performance, yet certain mid-sized models (such as 12B parameters) achieve comparable stability at a lower computational cost. This suggests that, in resource-constrained settings, carefully chosen mid-scale models may ofer an efective compromise.

Furthermore, a specific multilingual evaluation conducted with the SemEval development dataset shows that the narrative detection framework without prior training maintains promising performance in Bulgarian, Portuguese, Hindi, and Russian, indicating that the approach could be generalised beyond English and would support the broader applicability of the proposed methodology.

In conclusion, our findings validate the hypothesis that LLMs encode suficient pragmatic and contextual knowledge to identify strategic narratives in zero-shot settings. When guided by human-written coherent descriptions and combined through simple ensembling, these models rival supervised systems, ofering a scalable alternative for narrative detection in domains without training data.

![](images/f7469b68f760456bc9d1a18acbadc18e4ec7e9f77b9c0aa02709f851435d792b.jpg)  
Figure 7: Distribution of the number of input tokens per document for the OrigDesc prompt strategy on the SemEval dataset across the four evaluated models.

## 10. Future Work

An interesting extension of the present study would be to expand the analysis to multiple languages and investigate cross-lingual transfer. While the current work focuses primarily on English tweets and news articles, evaluating LLMs’ zero-shot narrative detection performance across languages could provide additional insights into their generalisation capabilities.

One promising direction is the incorporation of few-shot learning setups. While the current work is strictly zero-shot, our results suggest that modest gains could be achieved through minimal supervision. Given that the cost of producing highquality narrative annotations is relatively low compared to full dataset labelling, future research could explore the performance in few-shot scenarios, particularly in complex subnarrative detection tasks.

Another avenue involves the adoption of multi-agent architectures. Rather than relying on a single general-purpose model, future systems could benefit from ensembles of specialised agents, each prompted to focus on specific narrative features, such as stance, polarity, topical alignment, or temporal framing. These division of labour may facilitate more robust narrative identification, especially in multi-label and noisy environments.

Beyond these directions, future work could explore human-in-the-loop frameworks to support the generation of narrative descriptions. While automatic generation methods ofer scalability, they may lack alignment with annotators’ interpretative frameworks or the dataset’s underlying taxonomy. Introducing human supervision may improve semantic fidelity and task relevance.

In addition, a systematic analysis of instances consistently misclassified across models and runs could prove illuminating. These cases may reflect ambiguous or structurally atypical inputs that challenge current prompting paradigms. Understanding their distribution and linguistic characteristics may inform new strategies for model calibration, data augmentation, or annotation refinement.

## Acknowledgement

## References

[1] D. Sperber, D. Wilson, Relevance: Communication and cognition, Vol. 142, Harvard University Press Cambridge, MA, 1986.

[2] J. Dennison, Narratives: a review of concepts, determinants, efects, and uses in migration research, Comparative Migration Studies 9 (1) (2021) 50. doi:10.1186/s40878-021-00259-9. URL https://doi.org/10.1186/s40878-021-00259-9

[3] J. M. Adam, Les textes: types et prototypes, récit, description, argumentation, explication et dialogue, 4th Edition, Nathan, Paris, 2001.

[4] S. B. Chatman, Story and discourse: narrative structure in fiction and film, Cornell Paperbacks, Cornell Univ. Press, Ithaca, 2007.

[5] M. Toolan, Narrative: a critical linguistic introduction, Routledge, 2012.

[6] A. J. Greimas, Structural semantics: An attempt at a method, University of Nebraska Press, 1983.

[7] A. Piper, R. J. So, D. Bamman, Narrative theory for computational narrative understanding, in: M.-F. Moens, X. Huang, L. Specia, S. W.-t. Yih (Eds.), Proceedings of the 2021 conference on empirical methods in natural language

processing, Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 2021, pp. 298–311. doi:10.18653/v1/2021.emnlp-main.26. URL https://aclanthology.org/2021.emnlp-main.26/

[8] B. Kotseva, I. Vianini, N. Nikolaidis, N. Faggiani, K. Potapova, C. Gasparro, Y. Steiner, J. Scornavacche, G. Jacquet, V. Dragu, L. d. Rocca, S. Bucci, A. Podavini, M. Verile, C. Macmillan, J. P. Linge, Trend analysis of COVID-19 mis-/disinformation narratives–A 3-year study, Plos One 18 (11) (2023) e0291423. doi:10.1371/journal.pone.0291423. URL https://journals.plos.org/plosone/article?id=10.1371/journal. pone.0291423

[9] Y. Li, C. Scarton, X. Song, K. Bontcheva, Classifying COVID-19 vaccine narratives, in: R. Mitkov, G. Angelova (Eds.), Proceedings of the 14th international conference on recent advances in natural language processing, INCOMA Ltd., Shoumen, Bulgaria, Varna, Bulgaria, 2023, pp. 648–657. URL https://aclanthology.org/2023.ranlp-1.70/

[10] B. Hughes, C. Miller-Idriss, R. Piltch-Loeb, B. Goldberg, K. White, M. Criezis, E. Savoia, Development of a codebook of online anti-vaccination rhetoric to manage COVID-19 vaccine misinformation, International Journal of Environmental Research and Public Health 18 (14) (2021) 7556. doi:10.3390/ijerph18147556.

[11] T. G. Coan, C. Boussalis, J. Cook, M. O. Nanko, Computer-assisted classification of contrarian claims about climate change, Scientific Reports 11 (1) (2021) 22320, tex.copyright: 2021 The Author(s). doi:10.1038/s41598-021-01714-4. URL https://www.nature.com/articles/s41598-021-01714-4

[12] Tell us how you really feel: Analyzing pro-kremlin propaganda devices & narratives to identify sentiment implications. doi:10.13140/rg.2.2.15795.71209. URL https://www.researchgate.net/publication/368570477\_Tell\_Us\_ How\_You\_Really\_Feel\_Analyzing\_Pro-Kremlin\_Propaganda\_Devices\_ Narratives\_to\_Identify\_Sentiment\_Implications

[13] B. Santana, R. Campos, E. Amorim, A. Jorge, P. Silvano, S. Nunes, A survey on narrative extraction from textual data, Artificial Intelligence Review 56 (8) (2023) 8393–8435. doi:10.1007/s10462-022-10338-7. URL https://doi.org/10.1007/s10462-022-10338-7

[14] E. G. Mishler, Models of narrative analysis: a typology, Journal of Narrative and Life History 5 (2) (1995) 87–123. doi:10.1075/jnlh.5.2.01mod.

URL https://www.jbe-platform.com/content/journals/10.1075/jnlh.5. 2.01mod

[15] S. Wildemann, E. Elejalde, Automated identification of competing narratives in political discourse on social media.

[16] J. Elfes, Mapping news narratives using llms and narrative-structured text embeddings (Sep. 2024). doi:10.48550/arXiv.2409.06540. URL http://arxiv.org/abs/2409.06540

[17] E. Levi, G. Mor, T. Sheafer, S. Shenhav, Detecting narrative elements in informational text, in: Findings of the association for computational linguistics: NAACL 2022, Association for Computational Linguistics, Seattle, United States, 2022, pp. 1755–1765. doi:10.18653/v1/2022.findings-naacl.133. URL https://aclanthology.org/2022.findings-naacl.133

[18] F. Haouari, C. Scarton, N. Faggiani, N. Nikolaidis, B. Kotseva, I. A. Farha, J. Linge, K. Bontcheva, UKElectionNarratives: a dataset of misleading narratives surrounding recent UK general elections (May 2025). doi:10.48550/arXiv.2505.05459. URL http://arxiv.org/abs/2505.05459

[19] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, GPT-4 Technical Report, arXiv:2303.08774 [cs] (Mar. 2024). doi:10.48550/arXiv.2303.08774. URL http://arxiv.org/abs/2303.08774

[20] I. Singh, C. Scarton, K. Bontcheva, GateNLP at SemEval-2025 task 10: Hierarchical three-step prompting for multilingual narrative classification (May 2025). doi:10.48550/arXiv.2505.22867. URL http://arxiv.org/abs/2505.22867

[21] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of large language models (Oct. 2021). doi:10.48550/arXiv.2106.09685. URL http://arxiv.org/abs/2106.09685

[22] M. Eljadiri, D. Nurbakova, Team insalyon2 at SemEval-2025 task 10: a zeroshot agentic approach to text classification, in: S. Rosenthal, A. Rosá, D. Ghosh, M. Zampieri (Eds.), Proceedings of the 19th international workshop on semantic evaluation (SemEval-2025), Association for Computational Linguistics, Vienna,

Austria, 2025, pp. 965–980. URL https://aclanthology.org/2025.semeval-1.129/

[23] J. M. Fraile-Hernández, A. Peñas, UNEDTeam at SemEval-2025 task 10: Zeroshot narrative classification.

[24] J. M. Fraile-Hernández, A. Peñas, P. Moral, Automatic identification of narratives: Evaluation framework, annotation methodology, and dataset creation, IEEE access : practical innovations, open solutions 13 (2025) 11734–11753. doi:10.1109/access.2024.3475579. URL https://ieeexplore.ieee.org/document/10706846

[25] J. Piskorski, T. Mahmoud, N. Nikolaidis, R. Campos, A. Jorge, D. Dimitrov, P. Silvano, R. Yangarber, S. Sharma, T. Chakraborty, N. R. Guimarães, E. Sartori, N. Stefanovitch, Z. Xie, P. Nakov, G. D. S. Martino, SemEval-2025 task 10: Multilingual characterization and extraction of narratives from online news, in: Proceedings of the 19th international workshop on semantic evaluation, SemEval 2025, Vienna, Austria, 2025.

[26] M. Panahi, MaziyarPanahi/calme-2.4-rys-78b · hugging face (Nov. 2023). URL https://huggingface.co/MaziyarPanahi/calme-2.4-rys-78b

[27] Qwen Team, Qwen2.5 Technical ReportArXiv:2412.15115 [cs] (Jan. 2025). doi:10.48550/arXiv.2412.15115. URL http://arxiv.org/abs/2412.15115

[28] Gemma Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Rivière, L. Rouillard, T. Mesnard, G. Cideron, J.-b. Grill, S. Ramos, E. Yvinec, M. Casbon, E. Pot, I. Penchev, G. Liu, F. Visin, K. Kenealy, L. Beyer, X. Zhai, A. Tsitsulin, R. Busa-Fekete, A. Feng, N. Sachdeva, B. Coleman, Y. Gao, B. Mustafa, I. Barr, E. Parisotto, D. Tian, M. Eyal, C. Cherry, J.-T. Peter, D. Sinopalnikov, S. Bhupatiraju, R. Agarwal, M. Kazemi, D. Malkin, R. Kumar, D. Vilar, I. Brusilovsky, J. Luo, A. Steiner, A. Friesen, A. Sharma, A. Sharma, A. M. Gilady, A. Goedeckemeyer, A. Saade, A. Feng, A. Kolesnikov, A. Bendebury, A. Abdagic, A. Vadi, A. György, A. S. Pinto, A. Das, A. Bapna, A. Miech, A. Yang, A. Paterson, A. Shenoy, A. Chakrabarti, B. Piot, B. Wu, B. Shahriari, B. Petrini, C. Chen, C. L. Lan, C. A. Choquette-Choo, C. J. Carey, C. Brick, D. Deutsch, D. Eisenbud, D. Cattle, D. Cheng, D. Paparas, D. S. Sreepathihalli, D. Reid, D. Tran, D. Zelle, E. Noland, E. Huizenga, E. Kharitonov, F. Liu, G. Amirkhanyan,

G. Cameron, H. Hashemi, H. Klimczak-Plucińska, H. Singh, H. Mehta, H. T. Lehri, H. Hazimeh, I. Ballantyne, I. Szpektor, I. Nardini, J. Pouget-Abadie, J. Chan, J. Stanton, J. Wieting, J. Lai, J. Orbay, J. Fernandez, J. Newlan, J.-y. Ji, J. Singh, K. Black, K. Yu, K. Hui, K. Vodrahalli, K. Gref, L. Qiu, M. Valentine, M. Coelho, M. Ritter, M. Hofman, M. Watson, M. Chaturvedi, M. Moynihan, M. Ma, N. Babar, N. Noy, N. Byrd, N. Roy, N. Momchev, N. Chauhan, N. Sachdeva, O. Bunyan, P. Botarda, P. Caron, P. K. Rubenstein, P. Culliton, P. Schmid, P. G. Sessa, P. Xu, P. Stanczyk, P. Tafti, R. Shivanna, R. Wu, R. Pan, R. Rokni, R. Willoughby, R. Vallu, R. Mullins, S. Jerome, S. Smoot, S. Girgin, S. Iqbal, S. Reddy, S. Sheth, S. Põder, S. Bhatnagar, S. R. Panyam, S. Eiger, S. Zhang, T. Liu, T. Yacovone, T. Liechty, U. Kalra, U. Evci, V. Misra, V. Roseberry, V. Feinberg, V. Kolesnikov, W. Han, W. Kwon, X. Chen, Y. Chow, Y. Zhu, Z. Wei, Z. Egyed, V. Cotruta, M. Giang, P. Kirk, A. Rao, K. Black, N. Babar, J. Lo, E. Moreira, L. G. Martins, O. Sanseviero, L. Gonzalez, Z. Gleicher, T. Warkentin, V. Mirrokni, E. Senter, E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, Y. Matias, D. Sculley, S. Petrov, N. Fiedel, N. Shazeer, O. Vinyals, J. Dean, D. Hassabis, K. Kavukcuoglu, C. Farabet, E. Buchatskaya, J.-B. Alayrac, R. Anil, Dmitry, Lepikhin, S. Borgeaud, O. Bachem, A. Joulin, A. Andreev, C. Hardin, R. Dadashi, L. Hussenot, Gemma 3 Technical ReportArXiv:2503.19786 [cs] (Mar. 2025). doi:10.48550/arXiv.2503.19786. URL http://arxiv.org/abs/2503.19786

[29] LG AI Research, S. An, K. Bae, E. Choi, K. Choi, S. J. Choi, S. Hong, J. Hwang, H. Jeon, G. J. Jo, H. Jo, J. Jung, Y. Jung, H. Kim, J. Kim, S. Kim, S. Kim, S. Kim, Y. Kim, Y. Kim, Y. Kim, E. H. Lee, H. Lee, H. Lee, J. Lee, K. Lee, W. Lim, S. Park, S. Park, Y. Park, S. Yang, H. Yeen, H. Yun, EXAONE 3.5: Series of Large Language Models for Real-world Use CasesArXiv:2412.04862 [cs] (Dec. 2024). doi:10.48550/arXiv.2412.04862. URL http://arxiv.org/abs/2412.04862

[30] Ibm, Granite-3.3-2B-instruct (May 2025). URL https://huggingface.co/ibm-granite/granite-3.3-2b-instruct

[31] J. M. Fraile-Hernández, A. Peñas, On measuring large language models performance with inferential statistics, Information-an International Interdisciplinary Journal 16 (9) (2025) 817. doi:10.3390/info16090817. URL https://www.mdpi.com/2078-2489/16/9/817

[32] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal,

A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Wyatt, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Guzmán, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Thattai, G. Nail, G. Mialon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. Kloumann, I. Misra, I. Evtimov, J. Zhang, J. Copet, J. Lee, J. Gefert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. v. d. Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Prasad, K. Upasani, K. Plawiak, K. Li, K. Heafield, K. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, K. Lakhotia, L. Rantala-Yeary, L. v. d. Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher, L. Landzaat, L. d. Oliveira, M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Tsimpoukelli, M. Oldham, M. Rita, M. Pavlova, M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev, N. Chatterji, N. Zhang, O. Duchenne, O. Çelebi, P. Alrassy, P. Zhang, P. Li, P. Vasic, P. Weng, P. Bhargava, P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy, R. Calderer, R. S. Cabral, R. Stojnic, R. Raileanu, R. Maheswari, R. Girdhar, R. Patel, R. Sauvestre, R. Polidoro, R. Sumbaly, R. Taylor, R. Silva, R. Hou, R. Wang, S. Hosseini, S. Chennabasappa, S. Singh, S. Bell, S. S. Kim, S. Edunov, S. Nie, S. Narang, S. Raparthy, S. Shen, S. Wan, S. Bhosale, S. Zhang, S. Vandenhende, S. Batra, S. Whitman, S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom, T. Speckbacher, T. Mihaylov, T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet, V. Do, V. Vogeti, V. Albiero, V. Petrovic, W. Chu, W. Xiong, W. Fu, W. Meers, X. Martinet, X. Wang, X. Wang, X. E. Tan, X. Xia, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song, Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. Singh, A. Srivastava, A. Jain, A. Kelsey, A. Shajnfeld, A. Gangidi, A. Victoria, A. Goldstand, A. Menon, A. Sharma, A. Boesenberg, A. Baevski, A. Feinstein, A. Kallet, A. Sangani, A. Teo, A. Yunus, A. Lupu, A. Alvarado, A. Caples, A. Gu, A. Ho, A. Poulton, A. Ryan, A. Ramchandani, A. Dong, A. Franco, A. Goyal, A. Saraf, A. Chowdhury, A. Gabriel, A. Bharambe, A. Eisenman, A. Yazdan, B. James, B. Maurer,

B. Leonhardi, B. Huang, B. Loyd, B. D. Paola, B. Paranjape, B. Liu, B. Wu, B. Ni, B. Hancock, B. Wasti, B. Spence, B. Stojkovic, B. Gamido, B. Montalvo, C. Parker, C. Burton, C. Mejia, C. Liu, C. Wang, C. Kim, C. Zhou, C. Hu, C.-H. Chu, C. Cai, C. Tindal, C. Feichtenhofer, C. Gao, D. Civin, D. Beaty, D. Kreymer, D. Li, D. Adkins, D. Xu, D. Testuggine, D. David, D. Parikh, D. Liskovich, D. Foss, D. Wang, D. Le, D. Holland, E. Dowling, E. Jamil, E. Montgomery, E. Presani, E. Hahn, E. Wood, E.-T. Le, E. Brinkman, E. Arcaute, E. Dunbar, E. Smothers, F. Sun, F. Kreuk, F. Tian, F. Kokkinos, F. Ozgenel, F. Caggioni, F. Kanayet, F. Seide, G. M. Florez, G. Schwarz, G. Badeer, G. Swee, G. Halpern, G. Herman, G. Sizov, Guangyi, Zhang, G. Lakshminarayanan, H. Inan, H. Shojanazeri, H. Zou, H. Wang, H. Zha, H. Habeeb, H. Rudolph, H. Suk, H. Aspegren, H. Goldman, H. Zhan, I. Damlaj, I. Molybog, I. Tufanov, I. Leontiadis, I.-E. Veliche, I. Gat, J. Weissman, J. Geboski, J. Kohli, J. Lam, J. Asher, J.-B. Gaya, J. Marcus, J. Tang, J. Chan, J. Zhen, J. Reizenstein, J. Teboul, J. Zhong, J. Jin, J. Yang, J. Cummings, J. Carvill, J. Shepard, J. McPhie, J. Torres, J. Ginsburg, J. Wang, K. Wu, K. H. U, K. Saxena, K. Khandelwal, K. Zand, K. Matosich, K. Veeraraghavan, K. Michelena, K. Li, K. Jagadeesh, K. Huang, K. Chawla, K. Huang, L. Chen, L. Garg, L. A, L. Silva, L. Bell, L. Zhang, L. Guo, L. Yu, L. Moshkovich, L. Wehrstedt, M. Khabsa, M. Avalani, M. Bhatt, M. Mankus, M. Hasson, M. Lennie, M. Reso, M. Groshev, M. Naumov, M. Lathi, M. Keneally, M. Liu, M. L. Seltzer, M. Valko, M. Restrepo, M. Patel, M. Vyatskov, M. Samvelyan, M. Clark, M. Macey, M. Wang, M. J. Hermoso, M. Metanat, M. Rastegari, M. Bansal, N. Santhanam, N. Parks, N. White, N. Bawa, N. Singhal, N. Egebo, N. Usunier, N. Mehta, N. P. Laptev, N. Dong, N. Cheng, O. Chernoguz, O. Hart, O. Salpekar, O. Kalinli, P. Kent, P. Parekh, P. Saab, P. Balaji, P. Rittner, P. Bontrager, P. Roux, P. Dollar, P. Zvyagina, P. Ratanchandani, P. Yuvraj, Q. Liang, R. Alao, R. Rodriguez, R. Ayub, R. Murthy, R. Nayani, R. Mitra, R. Parthasarathy, R. Li, R. Hogan, R. Battey, R. Wang, R. Howes, R. Rinott, S. Mehta, S. Siby, S. J. Bondu, S. Datta, S. Chugh, S. Hunt, S. Dhillon, S. Sidorov, S. Pan, S. Mahajan, S. Verma, S. Yamamoto, S. Ramaswamy, S. Lindsay, S. Lindsay, S. Feng, S. Lin, S. C. Zha, S. Patil, S. Shankar, S. Zhang, S. Zhang, S. Wang, S. Agarwal, S. Sajuyigbe, S. Chintala, S. Max, S. Chen, S. Kehoe, S. Satterfield, S. Govindaprasad, S. Gupta, S. Deng, S. Cho, S. Virk, S. Subramanian, S. Choudhury, S. Goldman, T. Remez, T. Glaser, T. Best, T. Koehler, T. Robinson, T. Li, T. Zhang, T. Matthews, T. Chou, T. Shaked, V. Vontimitta, V. Ajayi, V. Montanez, V. Mohan, V. S. Kumar, V. Mangla, V. Ionescu, V. Poenaru, V. T. Mihailescu, V. Ivanov, W. Li, W. Wang, W. Jiang, W. Bouaziz, W. Con-

stable, X. Tang, X. Wu, X. Wang, X. Wu, X. Gao, Y. Kleinman, Y. Chen, Y. Hu, Y. Jia, Y. Qi, Y. Li, Y. Zhang, Y. Zhang, Y. Adi, Y. Nam, Yu, Wang, Y. Zhao, Y. Hao, Y. Qian, Y. Li, Y. He, Z. Rait, Z. DeVito, Z. Rosnbrick, Z. Wen, Z. Yang, Z. Zhao, Z. Ma, The llama 3 herd of models (Nov. 2024). doi:10.48550/arXiv.2407.21783.   
URL http://arxiv.org/abs/2407.21783

[33] A. Caballero, R. Centeno, Á. Rodrigo, LLM-based multi-agent models for multiclass classification of strategic narratives.

[34] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, W. E. Sayed, Mixtral of Experts, arXiv:2401.04088 [cs] (Jan. 2024). doi:10.48550/arXiv.2401.04088. URL http://arxiv.org/abs/2401.04088

[35] J. A. García-Díaz, R. Pan, A. R. Lázaro, C. Cristancho, R. Valencia-García, UMUTEAM at DIPROMATS 2024: Feature integration for detecting finegrained propaganda and narrative, in: CEUR workshop proceedings, Vol. 3756, 2024. URL https://portalinvestigacion.um.es/documentos/ 6702beb80194ce418010fb3f

[36] L. Tunstall, E. Beeching, N. Lambert, N. Rajani, K. Rasul, Y. Belkada, S. Huang, L. v. Werra, C. Fourrier, N. Habib, N. Sarrazin, O. Sanseviero, A. M. Rush, T. Wolf, Zephyr: Direct distillation of LM alignment (Oct. 2023). doi:10.48550/arXiv.2310.16944. URL http://arxiv.org/abs/2310.16944

[37] J. Tiedemann, S. Thottingal, OPUS-MT – Building open translation services for the World, in: A. Martins, H. Moniz, S. Fumega, B. Martins, F. Batista, L. Coheur, C. Parra, I. Trancoso, M. Turchi, A. Bisazza, J. Moorkens, A. Guerberof, M. Nurminen, L. Marg, M. L. Forcada (Eds.), Proceedings of the 22nd annual conference of the european association for machine translation, European Association for Machine Translation, Lisboa, Portugal, 2020, pp. 479–480. URL https://aclanthology.org/2020.eamt-1.61/

## Appendix A: Prompt Structure

To enhance clarity and interpretability, the prompts are colour-coded as follows:

• Blue: Instructional introduction or task formulation presented to the model.

• Green: Titles of narratives and sub-narratives.

• Grey: Descriptive content associated with each narrative or sub-narrative.

• Red: System-level instructions or control tokens, where applicable.

Your primary role is to analyse a tweet {tweet} and categorise them according to predefined narrative themes that reflect diferent portrayals and perspectives of China.

Your classification should help in understanding the overarching sentiments and strategic messaging in public discourse.

## Narratives to Detect:

## 1: The West as Immoral and Hostile:

Identify tweets that depict Western countries, especially the US, as immoral, hostile, or decadent. Look for content where China is positioned as a victim of Western actions or policies.

## 2: China as a Benevolent Power:

Classify tweets highlighting China’s peaceful and cooperative international stance. Include tweets that mention China’s contributions to global peace, support for international law, and economic development in other nations.

## 3: China’s Epic History:

Detect tweets that discuss China’s historical resilience and achievements, particularly those crediting the Chinese Communist Party with overcoming adversities and leading national modernisation.

## 4: China’s Political System and Values:

Look for tweets advocating for socialism with Chinese characteristics and portraying it as aligned with the will of the Chinese people. Tweets should suggest that China’s political system supports genuine democracy and global peace.

## 5: Success of the Chinese Communist Party’s Government:

Identify tweets focusing on the government’s role in driving China’s technological, economic, and social advancements, such as achievements in 5G technology.

## 6: China’s Cultural, Natural, and Heritage Appeal:

Classify tweets that promote Chinese culture, traditions, natural beauty, or heritage sites.   
This includes mentions of historical cities, cultural festivities, and natural landscapes.

## Instructions for Classification:

## 1. Read carefully the tweet {tweet}

2. Determine which narrative(s) it supports based on the content and sentiment expressed. A tweet may align with at most 2 narratives if it incorporates elements from more than one category. It is possible that the tweet does not support any narrative.

3. You have to generate a JSON structure:

"classification": [1, 2, 3, 4, 5, 6] at most 2 categories or []   
if the tweet doesn’t support any narrative,   
"reasoning": "reasoning of the answer in maximum 50 words."   
}