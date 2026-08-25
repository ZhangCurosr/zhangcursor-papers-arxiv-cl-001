# CARGO-T: Causal Reasoning Graph-of-Thought improves Multimodal Humor Comprehension

Abhilash Nandy Microsoft Research India

Rounak Saha Indian Institute of Science, India

Rahul Seetharaman LinkedIn, USA

Manav Nitin Kapadnis Apple, USA

Aman Bansal Nutanix, USA

Millon Madhur Das Fujitsu Research India

Pawan Goyal, Niloy Ganguly Indian Institute of Technology Kharagpur, India

## Abstract

In recent years, large-scale VLMs (Vision-Language Models) have achieved exceptional versatility and performance across a diverse array of tasks. However, these systems continue to face significant challenges in capturing and interpreting the subtle complexities of human humor, especially in a combination of both image and text modalities , as they involve complex, non-trivial interactions between people, objects, abstract concepts, and events, which constitute the foundational basis for a multitude of comedic mechanisms and humor signals. Causal graphs provide a natural formalism for representing the intertwined chains of events, entities, and contextual signals that give rise to humor in multimodal content. In this paper, we propose CARGO-T (Causal Reasoning Graph-of-Thought), which uses a graph serialized into a VLM-generated lightweight, code-based reasoning component that the same/different VLM can interpret to provide the final answer in a zero-shot/in-context learning setting. Through rigorous experimentation of CARGO-T on state-of-the-art commercial and open-source large VLMs on humor understanding and detection tasks—employing four datasets that span diverse comedic dimensions such as satire, sarcasm, and memes. CARGO-T improves performance by $\sim 1 - 2 0 \%$ in Humor Understanding and ∼ 1 − 3% in Humor Detection compared to other reasoning-based baselines. Further analysis on mutual information of the generated reasoning component reveals that CARGO-T offers more information compared to baselines that is relevant to the desired output<sup>1</sup>.

## 1 Introduction

“Analyzing humor is like dissecting a frog. Few people are interested and the frog dies of it.”

—E. B. White

Humor intricately weaves together the relationships among people, objects, concepts, and events, revealing subtle incongruities that both amuse and illuminate our understanding of everyday interactions. Effective humor comprehension similarly relies on advanced social reasoning and cultural literacy, as jokes often draw upon shared norms, stereotypes, and contextual cues to create meaning.

Additionally, humor can employ nonlinear narrative structures and symbolic incongruities—much like comics—demanding sophisticated reasoning to detect and appreciate the unexpected connections that make a joke resonate [37, 29]. Contemporary large-scale VLMs have demonstrated exceptional capabilities across a wide range of applications [26, 50, 46]. However, prior works which measure the ability of SOTA VLMs to understand humor in multimodal settings show that such VLMs fail at comprehending different kinds of humor such as satire, irony, sarcasm, etc. satisfactorily [32, 38, 20].

VLMs demonstrate significant shortcomings when humor derives from intricate social relations and object-mediated interactions. Such scenarios demand fine-grained modeling of affective cues and relational incongruities, which current architectures fail to represent adequately. For instance, in collaborative scenes requiring subtle appraisal of interpersonal dynamics and object-mediated interactions, current VLMs exhibit pronounced failures: they misclassify core emotional states (e.g., mistaking sarcasm for surprise) in dyadic exchanges and group settings [4], overlook incongruous action–object relationships that give rise to visual punchlines [16], and collapse multimodal figurative content into single literal interpretations, thereby missing alternate relational perspectives essential for nuanced humor [41]. Moreover, explicit reasoning augmentations fail to bridge these gaps: Chain-of-Thought [45] prompting suffers posterior collapse in subjective, emotion-laden contexts—retrieving static priors rather than dynamically reasoning over interpersonal cues—and yields negligible improvement on sarcasm and affect recognition benchmarks [11]. Likewise, multimodal extensions such as self-reflection frameworks [12] produce noisy, misaligned rationales that overlook the fine-grained affective and relational subtleties at the heart of social humor comprehension. Hence, there is a need to understand complex relations in scenarios involving interplay of multiple people, objects, concepts and/or events in a multimodal setting.

In order to understand and navigate through those intricacies, in this paper, we introduce causal reasoning graphs, a restricted subclass of causal graphs [14] that retains only cause–effect links—together with lightweight metadata describing objects, concepts, events, and participants—without any probabilistic parameters. These deterministic, event-centric structures serve as an interpretable backbone for modeling multifaceted multimodal scenarios. To the best of our knowledge, no prior work in multimodal humor comprehension has explicitly modeled causal graphs or cause–effect relationships to enhance humor understanding; we are the first to undertake this effort. (See Fig. 1 for an example of a causal reasoning graph)

![](images/f9fc60cea0f6bfac0b4c8c5cd20e8280a7dc50a12304c99482b1f46f56f67b53.jpg)  
Figure 1: Sample Causal Reasoning Graph for an image from the YesBut Dataset [32]. The image shows that in order to look fashionable, a person is wearing high heels, which in turn leads to discomfort. This is depicted using the cause-effect relations in the Causal Reasoning Graph, in addition to the description of objects (high heels, feet) and abstract concepts (fashion, discomfort) in the given image

In order to automatically construct these CRGs and subsequently use it

for humor detection and understanding task, we propose CARGO-T, which uses the power of VLM to first construct an explicit causal reasoning graph encoding the interplay of agents, objects, concepts, and events (along with relevant metadata) in the form of a lightweight, code-based reasoning script (Fig. 1 for reference). The causal reasoning graph is then ingested by the same/different VLM in a zero-shot/in-context-learning setting to detect and understand humor. The novelty of CARGO-T lies in using a causal reasoning graph as the reasoning component, instead of using natural language reasoning in Chain-of-Thought and Chain-of-Draft [45, 47], or using knowledge graph triplets in Zhao et al. [55]. The causal reasoning graph enables systematic causal traversal and compositional inference that accurately captures the dynamics and relational incongruities, which standard chain-of thought [45, 22] and other reasoning-based baselines [47, 31, 55] fail to capture.

Extensive evaluations on three benchmark datasets—spanning satire, sarcasm, and visual memes — demonstrate that CARGO-T consistently outperforms contemporary reasoning-based baselines by a significant margin of ∼ 1 − 20% in Humor Understanding and ∼ 1 − 3% in Humor Detection across different VLMs and settings. Furthermore, an information-theoretic analysis reveals that the generated reasoning component under CARGO-T contains more information than the baselines, and is also more relevant to the ground truth, offering stronger task-relevant supervision. Our key contributions are as follows: (i) we propose CARGO-T, a novel VLM-agnostic framework for constructing and leveraging explicit causal reasoning scripts in visual humor tasks; (ii) we show that CARGO-T enables systematic causal traversal and compositional inference beyond standard CoT and baseline methods; (iii) we conduct comprehensive experiments on four diverse humor datasets, achieving substantial performance gains; and (iv) we provide an in-depth mutual information analysis that quantifies the superiority of CARGO-T compared to baselines, and relevance of our reasoning component to the target outcome.

## 2 Related Work

LLMs and VLMs. Large-scale language and vision models have shown exceptional ability to follow human instructions and address a wide array of downstream tasks through zero-shot prompting [34, 2, 30, 50]. To systematically evaluate these advances, a variety of benchmarks have been developed, covering purely linguistic challenges [56, 13, 44, 17] as well as multimodal evaluations tailored for VLMs [51, 5, 6, 24, 23]. However, such models are still inadequate at understanding intricate social reasoning and human contexts [32, 4, 41, 11].

Humor Comprehension and AI. Humor serves as a fundamental aspect of human interaction [35], prompting extensive research into tasks like humor recognition [9, 7, 48] and humor generation [3]. Building on these foundations, recent studies have ventured into multimodal domains—forecasting visual humor [8], pinpointing humorous cartoon/meme captions [20, 15, 39], and detecting humor in video content [32, 28]. Nonetheless, despite these strides, LLMs such as ChatGPT are yet to fully conquer the complexities of computational humor [21].

Reasoning on Vision and Language. We focus on tasks related to humor comprehension in multimodal scenarios, where the model requires in-depth reasoning to comprehend the interactions between different parts of the image and text in the inputs. Prior Art has benchmarked reasoning capabilities of state-of-the-art models in tasks related to commonsense reasoning [6], visual question answering [18], and visio-linguistic compositionality [42].

Causal Reasoning and AI. A recent comprehensive survey [27] recasts the evaluation of large language models within a causal framework, examining their inferential strengths, strategies for mitigating bias and ensuring safety, methods for enriching outputs with interpretable explanations, and extensions into multimodal domains. Building on this causal perspective, Chen et al. [10] introduces a rigorous framework for assessing causal graph understanding in language models, grounded in four practical criteria drawn from philosophy and psychology. They further propose CLEAR, a benchmark on which LLMs exhibit early signs of causal comprehension, along with considerable room for improvement. In CausE [53], the focus pivots from traditional NLP tasks to the knowledge graph completion task, employing causal intervention and embedding disentanglement to mitigate confounders and achieve more stable predictions. CELLO [54] is a multimodal benchmark that evaluates VLMs on several multiple-choice causal questions. However, no prior art to our knowledge has utilized Cause-Effect Relations for boosting open-ended multimodal reasoning, which is the focus of this paper.

## 3 CARGO-T Framework

Given an image I and a task-specific text prompt P, the predicted answer is $\hat { \mathcal { Y } } \subseteq \mathcal { F } _ { \pmb { \theta } } ( \mathcal { T } , \mathcal { P } )$ . The VLM output $\mathcal { F } _ { \pmb { \theta } } ( \mathcal { I } , \mathcal { P } )$ contains a reasoning component ${ \mathcal { R } } ,$ followed by $\hat { \mathcal { V } } _ { : }$ , where $\mathcal { F } _ { \theta }$ represents the pre-trained VLM with parameters θ. This is a generalized representation of any reasoning-based method such as CoT, CoD etc. As a replacement we introduce CARGO-T a novel code-based reasoning framework using pre-trained VLMs on multimodal (vision-cum-image) inputs in a zeroshot/in-context learning setting to perform tasks related to humor comprehension.

## 3.1 CARGO-T: Zero-Shot Setting

The input prompt for zero-shot setting is given below for CARGO-T (task-specific placeholder is written in red). The prompt starts with the vanilla zero-shot task-specific query from prior art (e.g. For the Satirical Image Understanding Task evaluated in Nandy et al. [32], the vanilla zero-shot task-specific query used is - “Why is this image funny/satirical?”; all task-specific queries are listed in Section C ofAppendix). We then extend this base query by requesting the model to first construct a causal reasoning graph, represented in code, that captures the relationships among objects, people, and entities depicted in the image (and input text, if any, depending on the task), and subsequently produce the final answer for the task conditioned on the underlying cause-and-effect structure. By structuring the prompt into two distinct steps—graph generation followed by interpretation—we encourage the model to make its latent causal reasoning explicit before formulating the final answer.

## CARGO-T Zero-Shot Prompt

[VANILLA TASK-SPECIFIC QUERY]   
To answer this, first create a causal reasoning graph linking different objects, people, and entities present   
in the image (and input text, if any) in the form of a piece of code, and then give the final answer.   
Input Image: [INPUT IMAGE]   
Input Text: [INPUT TEXT]   
The output should be in the following format -   
Code: <Causal Reasoning Graph Code>   
Final Answer: <Final Answer>

Note that using CARGO-T in a zero-shot setting depends on the code-generation capability of the VLM, leveraging the VLM’s parametric knowledge. Based on the evaluation of code generated using VLMs in prior work [43], proprietary models such as GPT-4o and GPT-4o-mini [19] are far better at code generation in a zero-shot setting compared to open-source models. Hence, we mostly use closed-source proprietary VLMs for the CARGO-T framework in the zero-shot setting.

## 3.2 CARGO-T: In-Context Learning

CARGO-T in an in-context learning setting contains input-output pairs as in-context examples in the prompt, where the input consists of the input image and text (if any), and the output consists of the corresponding Causal Reasoning Graph followed by the answer. This is described in more detail as follows -

## 3.2.1 Curating In-Context Examples

Some dataset samples that are not part of the test set (e.g. from the training set) are used for curating the in-context examples. The input image (and text if any) and the ground truth final answer are available directly in the dataset sample. The corresponding Causal Reasoning Graph is curated by first generating a draft using the following zero-shot prompt (task and sample-specific placeholders are written in red), which generates said graph conditioned on the input and the final answer using GPT-4o.

## Prompt for generating an initial draft of a Causal Reasoning Graph for an in-context example

[VANILLA TASK-SPECIFIC QUERY]   
To answer this, first create a causal reasoning graph linking different objects, people, and entities present   
in the image (and input text, if any) in the form of a piece of code, and then give the final answer.   
Input Image: [INPUT IMAGE]   
Input Text: [INPUT TEXT]   
The output is -   
Code: <Causal Reasoning Graph Code>   
Final Answer: [GROUND TRUTH FINAL ANSWER]

These graphs are then rectified manually to adhere to a specific structure - starting by identifying the entities (an entity could either be an object, person, abstract concept, or event), listing the properties

of each entity, and finally listing the cause-effect relations, where each relation is a cause-effect pair derived from entities and their corresponding properties (see Fig. 1). An example of this manual rectification of causal reasoning graphs is shown in Fig. 3 in Section A of Appendix.

## 3.2.2 In-Context Learning Setup

CARGO-T: K-Shot In-Context Learning Prompt   
[VANILLA TASK-SPECIFIC QUERY]   
To answer this, first create a causal reasoning graph linking different objects, people, and entities present   
in the image (and input text, if any) in the form of a piece of code, and then give the final answer.   
## Example 1   
Input Image: [INPUT IMAGE #1]   
Input Text: [INPUT TEXT #1]   
The output is -   
Code: [CAUSAL REASONING GRAPH CODE #1]   
Final Answer: [GROUND TRUTH FINAL ANSWER #1]   
## Example #K   
// Exactly same format as Example #1   
## Example #K+1   
Input Image: [INPUT TEST IMAGE]   
Input Text: [INPUT TEST TEXT]   
The output should be in the following format -   
Code: <Causal Reasoning Graph Code>   
Final Answer: <Final Answer>  
For a test input, K in-context examples (in a K-shot setting) consisting of input images (and texts, if any), manually rectified causal graphs and the ground truth final answers and task-specific instructions are added to the prompt as shown above. With this prompt as input, The VLM is expected to generate a causal graph and the final answer corresponding to the test input at hand. Note that the VLM is not trained; it is used only for inference.

## 4 Experiments and Results

In this section, we first introduce the tasks and datasets used in our study, then describe the experimental setup and baseline methods, outline the evaluation metrics, present comparisons with those baselines, and conclude with an in-depth examination of CARGO-T’s reasoning component. In addition, we also perform an ablation study in Section E ofAppendix to show the importance of task-specific query and manually rectifying in-context examples in CARGO-T.

## 4.1 Tasks and Datasets

We focus on tasks related to Humor Understanding (answering why the input is funny) and Humor Detection (answering whether the input is funny or not) across different types of humor. Any such task can be formally written as in Section 3 - R is empty if there is no reasoning component. The ground truth answer Y for Humor Understanding is in natural language, while for Humor Detection, $\mathcal { V } \in \{ { } ^ {  } \mathrm { Y e s } ^ { , > } , { } ^ {  } \mathrm { N o } ^ { , > } \}$ and and the output $\hat { \mathcal { V } }$ should ideally be highly similar to Y. Examples of such tasks are shown in Fig. 4 of Section B in Appendix.

Tasks in Humor Understanding include - (1) Satirical Image Understanding: We evaluate a VLM’s satire understanding capability on satirical images of the YesBut Dataset [32] by prompting the VLMs to generate the corresponding punchlines. A holdout set of 5 diverse examples is manually selected for in-context example selection, and the rest of the 1, 079 satirical images are used for evaluation, (2) Meme Caption Generation: Given a meme image and title as input, the VLM needs to predict why that meme is funny. This is evaluated on 559 test samples from the MemeCap Dataset [20], and the in-context examples for in-context learning are manually selected from the corresponding training set.

Similarly, tasks in Humor Detection include - (1) Satirical Image Detection: This is a binary classification task in which the VLM must determine whether a given image is satirical or not. Similar to the Satirical Image Understanding Task, a holdout set of 6 (3 satirical, 3 non-satirical) samples is manually chosen for in-context example selection, and evaluation is done on 2, 541 (1, 081 satirical and 1, 460 non-satirical) images of the YesBut Dataset [32] (2) Multimodal Sarcasm Detection: This is also a binary classification task in which the VLM must determine whether a given image supplemented with a supporting text is sarcastic or not. This is evaluated on 2, 409 (1, 037 sarcastic and 1, 372 non-sarcastic) test samples of the MMSD 2.0 Dataset [38]. In-context examples for in-context learning are manually selected from the corresponding training set.

## 4.2 Experimental Setup

CARGO-T and the baselines are evaluated using the proprietary VLMs of GPT-4o and GPT-4o-mini [19] and the open-source MiniCPM<sup>2</sup> VLM [49]. The choice behind using these VLMs is inspired by the VLMs that are good at code generation as shown when evaluated on the Code-Vision Benchmark [43]. All experiments using open-source models are performed on 2 NVIDIA L40 GPUs, each having a VRAM of 48GB.

## 4.3 Baselines

CARGO-T is compared to the following VLM-agnostic baseline frameworks - (1) Vanilla. This baseline uses a prompt consisting of task-specific query and instructions. This is carried out in both zero-shot and few-shot in-context learning settings. In-Context Examples are added in the few-shot setting prompt (2) CoT (Chain-of-Thought) [45, 22] - CoT prompting asks the model to articulate intermediate reasoning steps before arriving at an answer, effectively turning black-box inference into an explicit, human-readable chain of deductions. Note that in the in-context learning setting, the rationale (reasoning component of CoT) of an in-context example is generated in a way similar to that in Section 3.2.1, except that the rationale is unstructured (see Section D for an example) (3) CoD (Chain-of-Draft) [47] - CoD is a prompting paradigm in which a model produces concise intermediate “draft” summaries of its reasoning—retaining only indispensable information—before arriving at a final answer, thereby reducing verbosity while preserving the logical structure. It is applied in a zero-shot setting (4) CCoT (Compositional Chain-of-Thought) [31] - CCoT is a zero-shot prompting technique that leverages automatically generated scene-graph representations to guide large multimodal models in capturing object attributes and inter-object relationships. By injecting the inferred scene graph into the prompt, it elicits structured, compositional reasoning without additional fine-tuning.

## 4.4 Evaluation Metrics

For the Humor Understanding Tasks, we compare the VLM generated text with the ground truth using automated text comparison metrics - lexical overlap using BLEU [36] and ROUGE-L [25], and semantic similarity using BERTScore [52], along with an “Avg. Score” that is the mean of these 3 metrics <sup>3</sup>. For the Humor Detection Tasks, we use binary classification evaluation metrics of Accuracy and macro-F1 Score.

## 4.5 CARGO-T on Humor Understanding

## 4.5.1 Zero-shot Setting

Table 1 shows the performance of CARGO-T vs. several baselines on the Satirical Image Understanding and Meme captioning Tasks in zero-shot setting across different VLMs. The corresponding percentage improvements of CARGO-T in comparison to the baselines are displayed in Fig. 2.

<table><tr><td rowspan="2">VLM</td><td rowspan="2">Method</td><td colspan="4">Satirical Image Understanding</td><td colspan="4">Meme Captioning</td></tr><tr><td>ROUGE-L</td><td>BLEU</td><td>BERTScore</td><td>Avg.</td><td>ROUGE-L</td><td>BLEU</td><td>BERTScore</td><td>Avg.</td></tr><tr><td rowspan="5">MiniCPM (0-shot)</td><td>Vanilla</td><td>0.1669</td><td>0.0108</td><td>0.8589</td><td>0.3455</td><td>0.0789</td><td>0.0073</td><td>0.833</td><td>0.3064</td></tr><tr><td>CoT</td><td>0.163</td><td>0.0155</td><td>0.8586</td><td>0.3457</td><td>0.0646</td><td>0.0033</td><td>0.8303</td><td>0.2994</td></tr><tr><td>CoD</td><td>0.1684</td><td>0.0137</td><td>0.8616</td><td>0.3479</td><td>0.0888</td><td>0.0059</td><td>0.8394</td><td>0.3114</td></tr><tr><td>CCoT</td><td>0.1482</td><td>0.0086</td><td>0.8541</td><td>0.337</td><td>0.0744</td><td>0.0040</td><td>0.8404</td><td>0.3063</td></tr><tr><td>CARGo-T</td><td>0.1779</td><td>0.0139</td><td>0.8594</td><td>0.3504</td><td>0.126</td><td>0.0123</td><td>0.8503</td><td>0.3295</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">GPT-4o-mini (0-shot)</td><td>Vanilla</td><td>0.1493</td><td>0.0098</td><td>0.8525</td><td>0.3372</td><td>0.0889</td><td>0.0045</td><td>0.845</td><td>0.3128</td></tr><tr><td>CoT</td><td>0.088</td><td>0.0052</td><td>0.8197</td><td>0.3043</td><td>0.1178</td><td>0.0076</td><td>0.8453</td><td>0.3236</td></tr><tr><td>CoD</td><td>0.1388</td><td>0.0107</td><td>0.834</td><td>0.3278</td><td>0.1282</td><td>0.0078</td><td>0.8459</td><td>0.3273</td></tr><tr><td>CCoT</td><td>0.1184</td><td>0.0062</td><td>0.8402</td><td>0.3216</td><td>0.1315</td><td>0.0069</td><td>0.8566</td><td>0.3317</td></tr><tr><td>CARGO-T</td><td>0.2024</td><td>0.0185</td><td>0.8687</td><td>0.3632</td><td>0.1377</td><td>0.008</td><td>0.8497</td><td>0.3318</td></tr><tr><td>GPT-40</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">(0-shot)</td><td>Vanilla</td><td>0.1893</td><td>0.0152</td><td>0.8667</td><td>0.3571</td><td>0.1092</td><td>0.006</td><td>0.8512</td><td>0.3221</td></tr><tr><td>CoT</td><td>0.1459</td><td>0.0082</td><td>0.847</td><td>0.3337</td><td>0.1183</td><td>0.0064</td><td>0.8523</td><td>0.3257</td></tr><tr><td>CoD</td><td>0.2064</td><td>0.02</td><td>0.8725</td><td>0.3663</td><td>0.1264</td><td>0.0069</td><td>0.8535</td><td>0.3289</td></tr><tr><td>CCoT</td><td>0.1605</td><td>0.0094</td><td>0.8564</td><td>0.3421</td><td>0.1055</td><td>0.0057</td><td>0.8535</td><td>0.3216</td></tr><tr><td>CARGo-T</td><td>0.2219</td><td>0.0245</td><td>0.8715</td><td>0.3726</td><td>0.1316</td><td>0.0078</td><td>0.8569</td><td>0.3321</td></tr></table>

Table 1: CARGO-T vs. Baselines for Satirical Image Understanding Task on YesBut Dataset and Meme Captioning Task on MemeCap Dataset in zero-shot setting. The best and second-best results for a VLM are highlighted in bold and italic respectively

We observe that - (1) Given MiniCPM as the VLM, CARGO-T gives the best Avg. Score, boosting the Avg. Score by 0.72% and 5.81% compared to the best baseline on the Satire and Meme Understanding Tasks respectively<sup>4</sup>, suggesting that CARGO-T is useful for improving humor comprehension of small open-source VLMs (2) CARGO-T gives a consistent improvement across all baselines when using GPT-4o-mini as the VLM (unlike MiniCPM), which could be attributed to superior code generation and understanding capabilities compared to MiniCPM (as CARGO-T uses Code-based reasoning) due to a larger context window length (GPT-4o-mini: 128k tokens, MiniCPM: 32k tokens), even though both models have similar sizes [49, 1] (3) When using GPT-4o as the

![](images/4f1ffff2e39a74a0cd3a7d8afa10c32ad2e5723a5ee0f6cd73784e8fdfe6f796.jpg)  
Figure 2: Percentage Improvement of CARGO-T compared to Baselines in zero-shot setting when evaluating on Satirical Image Understanding

VLM, CARGO-T is the best compared to baselines across the tasks, and also is better than when using GPT-4o-mini and MiniCPM, which could be attributed to the larger model size of GPT-4o (4) According to Fig. 2, CARGO-T gives the highest performance improvement compared to the best baseline on the YesBut Dataset when using GPT-4o-mini VLM, reinforcing the notion that CARGO-T is effective even when the model is small.

## 4.5.2 Few-Shot In-Context Learning Setting

Table 2 shows the performance of CARGO-T vs. several baselines on the Satirical Image Understanding and Meme captioning Tasks in few-shot setting across different number of in-context examples and VLMs. We observe that - (1) CARGO-T outperforms the baselines across all metrics in 2-shot setting, suggesting that a little in-context supervision is sufficient for effective causal reasoning graph generation across VLMs (2) CARGO-T gives the best Avg. Score when using the GPT-4o VLM, which is again due to its larger size. (3) When increasing the in-context examples, there is a diminishing return in performance improvement (e.g. CARGO-T gives a performance improvement of 11.66% in 0-shot, 10.14% in 2-shot, and 5.86% in 5-shot setting compared to CoT when using GPT-4o VLM), suggesting that increasing in-context examples does not necessarily improve VLM’s understanding capability. (4) As model size increases, having more in-context examples does not necessarily translate to better performance of CARGO-T, probably due to better parametric knowledge of the model (5) CARGO-T gives better performance improvement on Meme Captioning compared

to Satire Understanding, suggesting that a supporting text (here, meme title) as an input in addition to the image in in-context examples leads to better supervision.
<table><tr><td rowspan="2">VLM</td><td rowspan="2">Method</td><td colspan="4">Satirical Image Understanding</td><td colspan="4">Meme Captioning</td></tr><tr><td>ROUGE-L</td><td>BLEU</td><td>BERTScore</td><td>Avg.</td><td>ROUGE-L</td><td>BLEU</td><td>BERTScore</td><td>Avg.</td></tr><tr><td rowspan="3">MiniCPM (2-shot)</td><td>Vanilla</td><td>0.2187</td><td>0.0251</td><td>0.8763</td><td>0.3734</td><td>0.0832</td><td>0.0078</td><td>0.8365</td><td>0.3092</td></tr><tr><td>CoT</td><td>0.2198</td><td>0.0278</td><td>0.8778</td><td>0.3751</td><td>0.1078</td><td>0.0099</td><td>0.8463</td><td>0.3213</td></tr><tr><td>CARGo-T</td><td>0.2274</td><td>0.0305</td><td>0.8791</td><td>0.379</td><td>0.134</td><td>0.0136</td><td>0.8541</td><td>0.3339</td></tr><tr><td rowspan="3">MiniCPM (5-shot)</td><td>Vanilla</td><td>0.2508</td><td>0.0351</td><td>0.8861</td><td>0.3907</td><td>0.0839</td><td>0.0079</td><td>0.837</td><td>0.3096</td></tr><tr><td>CoT</td><td>0.2404</td><td>0.035</td><td>0.8848</td><td>0.3867</td><td>0.1183</td><td>0.0106</td><td>0.8471</td><td>0.3253</td></tr><tr><td>CARGo-T</td><td>0.2414</td><td>0.0368</td><td>0.8851</td><td>0.3878</td><td>0.1353</td><td>0.0138</td><td>0.8546</td><td>0.3346</td></tr><tr><td rowspan="3">GPT-4o-mini</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>0.1837</td><td>0.0148</td><td>0.8654</td><td>0.3546</td><td>0.0935</td><td>0.0047</td><td>0.8467</td><td>0.315</td></tr><tr><td>CoT</td><td>0.2068</td><td>0.0242</td><td>0.8729</td><td>0.368</td><td>0.1092</td><td>0.0068</td><td>0.8497</td><td>0.3219</td></tr><tr><td rowspan="4">(2-shot) GPT-4o-mini (5-shot)</td><td>CARGO-T</td><td>0.2384</td><td>0.0325</td><td>0.8814</td><td>0.3841</td><td>0.1267</td><td>0.0071</td><td>0.8573</td><td>0.3304</td></tr><tr><td>Vanilla</td><td>0.218</td><td>0.0235</td><td>0.8773</td><td>0.3729</td><td>0.0937</td><td>0.0049</td><td>0.8471</td><td>0.3152</td></tr><tr><td>CoT</td><td>0.2289</td><td>0.0296</td><td>0.8822</td><td>0.3802</td><td>0.1053</td><td>0.0074</td><td>0.8543</td><td>0.3223</td></tr><tr><td>CARGO-T</td><td>0.2396</td><td>0.0294</td><td>0.8567</td><td>0.3752</td><td>0.13</td><td>0.0076</td><td>0.8584</td><td>0.332</td></tr><tr><td rowspan="3">GPT-40</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>0.2146</td><td>0.0244</td><td>0.8415</td><td>0.3602</td><td>0.1153</td><td>0.0065</td><td>0.8531</td><td>0.325</td></tr><tr><td>CoT</td><td>0.1831</td><td>0.0141</td><td>0.8681</td><td>0.3551</td><td>0.126</td><td>0.0083</td><td>0.8535</td><td>0.3293</td></tr><tr><td rowspan="3">(2-shot) GPT-40 (5-shot)</td><td>CARGo-T</td><td>0.2534</td><td>0.0339</td><td>0.886</td><td>0.3911</td><td>0.1479</td><td>0.0095</td><td>0.8636</td><td>0.3403</td></tr><tr><td>Vanilla</td><td>0.231</td><td>0.0266</td><td>0.8417</td><td>0.3664</td><td>0.1188</td><td>0.0068</td><td>0.8543</td><td>0.3266</td></tr><tr><td>CoT CARGO-T</td><td>0.206 0.2513</td><td>0.0198 0.0318</td><td>0.8797 0.8872</td><td>0.3685 0.3901</td><td>0.1315 0.1624</td><td>0.009 0.0106</td><td>0.8578 0.8654</td><td>0.3328 0.3461</td></tr></table>

Table 2: CARGO-T vs. Baselines for Satirical Image Understanding Task on YesBut Dataset and Meme Captioning Task on MemeCap Dataset in few-shot setting. The best and second-best results for a VLM and a particular number of in-context examples are highlighted in bold and italic respectively

## 4.6 CARGO-T on Humor Detection

<table><tr><td>Metric</td><td>Vanilla</td><td>CoT</td><td>CARGO-T</td><td>Impr.</td></tr><tr><td colspan="5">0-shot</td></tr><tr><td>Accuracy</td><td>47.42%</td><td>48.07%</td><td>49.48%</td><td>2.93%</td></tr><tr><td>F1 Score</td><td>61.05%</td><td>61.69%</td><td>62.20%</td><td>0.83%</td></tr><tr><td colspan="5">2-shot</td></tr><tr><td>Accuracy</td><td>47.81%</td><td>48.32%</td><td>49.88%</td><td>3.23%</td></tr><tr><td>F1 Score</td><td>61.22%</td><td>61.73%</td><td>62.38%</td><td>1.05%</td></tr><tr><td colspan="5">6-shot</td></tr><tr><td>Accuracy</td><td>47.85%</td><td>48.61%</td><td>49.91%</td><td>2.67%</td></tr><tr><td>F1 Score</td><td>61.27%</td><td>61.85%</td><td>62.41%</td><td>0.9%</td></tr></table>

Table 3: Comparison of Vanilla, CoT and CARGO-T across different number of in-context examples on Sarcasm Detection in MMSD 2.0 Dataset using GPT-4o, along with performance metric improvements compared to best baseline

<table><tr><td>Metric</td><td>Vanilla</td><td>CoT</td><td>CARGO-T</td><td>Impr.</td></tr><tr><td colspan="5">0-shot</td></tr><tr><td>Accuracy</td><td>42.60%</td><td>42.7%</td><td>43.18%</td><td>1.12%</td></tr><tr><td>F1 Score</td><td>59.69%</td><td>59.75%</td><td>59.97%</td><td>0.37%</td></tr><tr><td colspan="5">2-shot</td></tr><tr><td>Accuracy</td><td>44.05%</td><td>44.39%</td><td>44.63%</td><td>0.54%</td></tr><tr><td>F1 Score</td><td>60.52%</td><td>60.68%</td><td>61.01%</td><td>0.54%</td></tr><tr><td colspan="5">6-shot</td></tr><tr><td>Accuracy</td><td>44.91%</td><td>45.38%</td><td>45.57%</td><td>1.05%</td></tr><tr><td>F1 Score</td><td>61.08%</td><td>61.32%</td><td>61.56%</td><td>0.39%</td></tr></table>

Table 4: Comparison of Vanilla, CoT, and CARGO-T across different number of in-context examples on Satire Detection in YesBut Dataset using GPT-4o, along with performance metric improvements compared to best baseline

Tables 3 and 4 show the performance improvement compared to the best baseline when using CARGO-T on GPT-4o for Sarcasm Detection on the MMSD 2.0 Dataset and Satire Detection on the YesBut Dataset respectively. Note that intuitively, Humor Detection is easier than Humor Understanding - hence, we compare CARGO-T with Vanilla and the well-performing baseline of CoT for experiments on Humor Detection. We observe that CARGO-T performs consistently better across varying number of in-context examples. In addition, the performance improvement in detecting sarcasm is better than satire, which might be due to the additional supporting text in the sarcasm detection task providing additional supervision while generating a causal reasoning graph.

## 4.7 Dissecting CARGO-T’s Reasoning Component

To investigate the underlying factors contributing to the superior performance of CARGO-T, we specifically analyze the reasoning component (R) generated by CARGO-T in comparison to that produced by baseline methods. Our analysis focuses on two key aspects: (A) Whether the reasoning produced by CARGO-T contains richer and more novel information relative to the baselines, and (B) Whether the final ground truth can be more effectively inferred from the reasoning provided by CARGO-T than from that of the baselines.

To assess (A), we employ pairwise dissimilarity/divergence based measures across CARGO-T and baselines - KL Divergence between the token distributions for measuring the extra amount of lexical information in R of one method compared to another, and a sentence similarity-based dis-similarity score for measuring the amount of semantically newer information in a method’s R relative to another. Similarly, to assess (B), we employ an LLM-as-a-judge [56] approach to infer whether Y can be logically inferred from R. To ensure uniform evaluation, we apply CARGO-T and the baselines of CoT, CoD, and CCoT using MiniCPM in zero-shot setting to the Satirical Image Understanding Task on the YesBut Dataset. These are described in detail as follows -

## 4.7.1 KL Divergence between token distributions

Given the texts $T _ { 1 } , T _ { 2 } , K L ( T _ { 1 } | | T _ { 2 } )$ and $K L ( T _ { 2 } | | T _ { 1 } )$ are calculated after tokenizing $T _ { 1 }$ and $T _ { 2 }$ by splitting with whitespace and making the tokens lowercase, and the set of all tokens so obtained from $T _ { 1 }$ and $T _ { 2 }$ forms the vocabulary (check Algorithm 1 in Section F ofAppendix for details). Since KL-divergence measures how well one distribution approximates another, a higher $K L ( T _ { 1 } | | T _ { 2 } )$ than $K L ( T _ { 2 } | | \mathbf { \breve { { T } } } _ { 1 } )$ suggests that the lexical distribution in $T _ { 1 }$ deviates more from $\breve { T } _ { 2 } \ { ^ \bullet }$ distribution than vice versa, possibly indicating that $T _ { 1 }$ contains more varied or less predictable lexical content. Table 5 shows the KL-Divergence between reasoning components of CARGO-T and baselines. We see that CARGO-T has a greater or equal amount of newer lexical information compared to the 3 baselines.

<table><tr><td> $\overline { { T _ { 1 } } }$ </td><td> $\overline { { T _ { 2 } } }$ </td><td> $\overline { { K L ( T 1 | | T 2 ) } }$ </td></tr><tr><td> $\overline { { \mathcal { R } _ { \mathrm { C A R G o - T } } } }$   $\mathcal { R } _ { C o T }$ </td><td> $\overline { { \mathcal { R } _ { C o T } } }$ </td><td>0.21 0.19</td></tr><tr><td> $\overline { { \mathcal { R } _ { \mathrm { C A R G o - T } } } }$ </td><td> ${ \underline { { \mathcal { R } _ { \mathrm { C A R G o - T } } } } }$   $\overline { { \mathcal { R } \ v { } _ { C o D } } }$ </td><td>0.21</td></tr><tr><td> $\mathcal { R } _ { C o D }$ </td><td> $\mathcal { R } _ { \mathrm { C A R G o - T } }$ </td><td>0.20</td></tr><tr><td> $\overline { { \mathcal { R } _ { \mathrm { C A R G o - T } } } }$ </td><td> $\scriptstyle { \mathcal { R } } _ { C C o T }$ </td><td>0.25</td></tr><tr><td> $\scriptstyle { \mathcal { R } } _ { C C o T }$ </td><td> $\mathcal { R } _ { \mathrm { C A R G o - T } }$ </td><td>0.25</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 5: KL-Divergence between methods (Method name is in subscript)

<table><tr><td rowspan=1 colspan=1> $\overline { { T _ { 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { T _ { 2 } } }$ </td><td rowspan=1 colspan=1> $\overline { { L S F ( T _ { 1 } | | T _ { 2 } ) } }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { R } _ { \mathbf { C } _ { \mathrm { A R G O - T } } } } }$  $\scriptstyle { \mathcal { R } } _ { C o T }$ </td><td rowspan=1 colspan=1> $\scriptstyle { \mathcal { R } } _ { C o T }$  $\mathcal { R } _ { \mathrm { C A R G o - T } }$ </td><td rowspan=1 colspan=1>10.85</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { R } _ { \mathrm { C A R G o - T } } } }$  $\underline { { \mathcal { R } _ { C o D } } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { R } \ v { } _ { C o D } } }$  $\mathcal { R } _ { \mathrm { C A R G o - T } }$ </td><td rowspan=1 colspan=1>10.84</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { R } _ { \mathrm { C A R G o - T } } } }$  $\scriptstyle { \mathcal { R } } _ { C C o T }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { R } _ { C C o T } } }$  $\mathcal { R } _ { \mathrm { C A R G o - T } }$ </td><td rowspan=1 colspan=1>0.980.94</td></tr></table>

Table 6: LSF between methods (Method name is in subscript)

## 4.7.2 Sentence similarity-based dissimilarity score

In sentence-based similarity, denoted as $L S F ( T _ { y } \parallel T _ { x } )$ , we compute the fraction of sentences in $T _ { x }$ that are semantically similar to at least one sentence in $T _ { y } .$ . This metric reflects the extent to which the content of $T _ { x }$ is covered by $T _ { y }$ . Sentence similarity is evaluated using Sentence-BERT [40], where two sentences are considered similar if the cosine similarity between their respective embeddings exceeds 0.5 (refer to Algorithm 2 in Section F of the Appendix for details). Similar to KL-Divergence, if $L S F ( T _ { 1 } | | T _ { 2 } ) > L S F ( T _ { 2 } | | T _ { 1 } )$ , the additional semantic information in $T _ { 1 }$ is greater than that in $T _ { 2 }$ . Table 6 shows LSF between reasoning components of CARGO-T and baselines. We can see that CARGO-T has a greater amount of newer semantic information compared to the 3 baselines.

## 4.7.3 LLM-as-a-judge approach to infer whether Y can be logically inferred from R

A detailed prompt containing R and Y (as well as details of the Humor Comprehension task) is given as input to GPT-4 [33], and a prediction of 1 means that Y can be logically inferred from $\mathcal { R } _ { : }$ , and 0 means otherwise (here, we rely on the parametric knowledge of GPT-4). Given a method, this is done for all dataset samples and the percentage of samples giving a prediction of 1 is a score (referred to here as INFERSCORE) representing the effectiveness of the generated

Table 7: INFERSCORE: CARGO-T vs. Baselines
<table><tr><td>Method</td><td>INFERSCORE</td></tr><tr><td>CoT</td><td>40.78</td></tr><tr><td>CoD</td><td>40.68</td></tr><tr><td>CCoT</td><td>37.64</td></tr><tr><td>CARGO-T</td><td>45.11</td></tr></table>

reasoning component of the method. Table 7 shows that CARGO-T attains the highest INFERSCORE.

## 5 Conclusion

This work shows the effectiveness of using VLMs to generate Causal Reasoning Graphs, thereby enhancing Multimodal Humor Comprehension. Our approach CARGO-T leverages causal reasoning graphs to systematically model the relationships among events, entities, and contextual cues in multimodal inputs. Experiments validate the superiority of CARGO-T across VLMs in comparison to other reasoning-based baselines such as CoT, CoD, and CCoT in both zero-shot and few-shot in-context learning settings. Notably, Causal Reasoning Graphs enhance the amount of relevant reasoning information that is required to better understand and detect humor in multimodal scenarios.

## References

[1] Asma Ben Abacha, Wen-wai Yim, Yujuan Fu, Zhaoyi Sun, Meliha Yetisgen, Fei Xia, and Thomas Lin. Medec: A benchmark for medical error detection and correction in clinical notes. arXiv preprint arXiv:2412.19260, 2024.

[2] AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/ blob/main/MODEL\_CARD.md.

[3] Miriam Amin and Manuel Burghardt. A survey on approaches to computational humor generation. In Stefania DeGaetano, Anna Kazantseva, Nils Reiter, and Stan Szpakowicz, editors, Proceedings of the 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature, pages 29–41, Online, December 2020. International Committee on Computational Linguistics. URL https: //aclanthology.org/2020.latechclfl-1.4.

[4] Sree Bhattacharyya and James Z Wang. Evaluating vision-language models for emotion recognition. arXiv preprint arXiv:2502.05660, 2025.

[5] Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, and Ludwig Schimdt. Visit-bench: a benchmark for vision-language instruction following inspired by real-world use. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 26898–26922, 2023.

[6] Nitzan Bitton-Guetta, Yonatan Bitton, Jack Hessel, Ludwig Schmidt, Yuval Elovici, Gabriel Stanovsky, and Roy Schwartz. Breaking common sense: Whoops! a vision-and-language benchmark of synthetic and compositional images. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2616–2627, 2023.

[7] Andrew Cattle and Xiaojuan Ma. Recognizing humour using word associations and humour anchor extraction. In Proceedings of the 27th international conference on computational linguistics, pages 1849–1858, 2018.

[8] Arjun Chandrasekaran, Ashwin K Vijayakumar, Stanislaw Antol, Mohit Bansal, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. We are humor beings: Understanding and predicting visual humor. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4603–4612, 2016.

[9] Lei Chen and Chong MIn Lee. Predicting audience’s laughter using convolutional neural network. arXiv preprint arXiv:1702.02584, 2017.

[10] Sirui Chen, Mengying Xu, Kun Wang, Xingyu Zeng, Rui Zhao, Shengjie Zhao, and Chaochao Lu. Clear: Can language models really understand causal graphs? In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 6247–6265, 2024.

[11] Georgios Chochlakis, Niyantha Maruthu Pandiyan, Kristina Lerman, and Shrikanth Narayanan. Larger language models don’t care how you think: Why chain-of-thought prompting fails in subjective tasks. arXiv preprint arXiv:2409.06173, 2024.

[12] Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, Quanquan Gu, James Y Zou, Kai-Wei Chang, and Wei Wang. Enhancing large vision language models with self-training on image comprehension. Advances in Neural Information Processing Systems, 37:131369–131397, 2024.

[13] Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36, 2024.

[14] Malte Helmert. A planning heuristic based on causal graph analysis. In ICAPS, volume 16, pages 161–170, 2004.

[15] Jack Hessel, Ana Marasovic, Jena D. Hwang, Lillian Lee, Jeff Da, Rowan Zellers, Robert Mankoff, and Yejin Choi. Do androids laugh at electric sheep? humor “understanding” benchmarks from the new yorker caption contest. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 688–714, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.41. URL https://aclanthology.org/2023.acl-long.41.

[16] Zhe Hu, Tuo Liang, Jing Li, Yiren Lu, Yunlai Zhou, Yiran Qiao, Jing Ma, and Yu Yin. Cracking the code of juxtaposition: Can AI models understand the humorous contradictions. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=bCMpdaQCNW.

[17] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems, 36, 2024.

[18] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.

[19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[20] EunJeong Hwang and Vered Shwartz. MemeCap: A dataset for captioning and interpreting memes. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1433–1445, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.89. URL https://aclanthology.org/2023.emnlp-main.89.

[21] Sophie Jentzsch and Kristian Kersting. Chatgpt is fun, but it is not funny! humor is still challenging large language models. arXiv preprint arXiv:2306.04563, 2023.

[22] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

[23] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

[24] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024.

[25] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013/.

[26] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.

[27] X Liu, P Xu, J Wu, J Yuan, Y Yang, Y Zhou, F Liu, T Guan, H Wang, T Yu, et al. Large language models and causal inference in collaboration: A comprehensive survey. arxiv. 2024. arXiv preprint arXiv:2403.09606.

[28] Yang Liu, Tongfei Shen, Dong Zhang, Qingying Sun, Shoushan Li, and Guodong Zhou. Comment-aided video-language alignment via contrastive pre-training for short-form video humor detection. arXiv preprint arXiv:2402.09055, 2024.

[29] Alan D Manning. Understanding comics: The invisible art. 1998.

[30] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. arXiv preprint arXiv:2402.06196, 2024.

[31] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-ofthought prompting for large multimodal models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14420–14431, 2024.

[32] Abhilash Nandy, Yash Agarwal, Ashish Patwa, Millon Madhur Das, Aman Bansal, Ankit Raj, Pawan Goyal, and Niloy Ganguly. \*\*\*YesBut\*\*\*: A high-quality annotated multimodal dataset for evaluating satire comprehension capability of vision-language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 16878–16895, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. emnlp-main.937. URL https://aclanthology.org/2024.emnlp-main.937/.

[33] OpenAI. Gpt-4 technical report, 2023.

[34] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[35] Jerry Palmer. Taking humour seriously. Routledge, 2003.

[36] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040.

[37] Jessica Pressman. Digital modernism: Making it new in new media. Oxford University Press, USA, 2014.

[38] Libo Qin, Shijue Huang, Qiguang Chen, Chenran Cai, Yudi Zhang, Bin Liang, Wanxiang Che, and Ruifeng Xu. Mmsd2. 0: Towards a reliable multi-modal sarcasm detection system. In Findings of the Association for Computational Linguistics: ACL 2023, pages 10834–10845, 2023.

[39] Dragomir Radev, Amanda Stent, Joel Tetreault, Aasish Pappu, Aikaterini Iliakopoulou, Agustin Chanfreau, Paloma de Juan, Jordi Vallmitjana, Alejandro Jaimes, Rahul Jha, et al. Humor in collective discourse: Unsupervised funniness detection in the new yorker cartoon caption contest. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 475–479, 2016.

[40] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, 2019.

[41] Arkadiy Saakyan, Shreyas Kulkarni, Tuhin Chakrabarty, and Smaranda Muresan. Understanding figurative meaning through explainable visual entailment. arXiv preprint arXiv:2405.01474, 2024.

[42] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238–5248, 2022.

[43] Hao Wang, Xiaotian Zhou, Zheng Xu, Kaixin Cheng, Yizhou Zuo, Kai Tian, Jiaming Song, Jingren Lu, Wei Hu, and Xiaodong Liu. Code-Vision: Evaluating multimodal llms logic understanding and code generation capabilities. arXiv preprint arXiv:2502.11829, feb 2025.

[44] Yidong Wang, Zhuohao Yu, Wenjin Yao, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, et al. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. In ICLR, 2024.

[45] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf.

[46] Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and S Yu Philip. Multimodal large language models: A survey. In 2023 IEEE International Conference on Big Data (BigData), pages 2247–2256. IEEE, 2023.

[47] Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by writing less. arXiv preprint arXiv:2502.18600, 2025.

[48] Diyi Yang, Alon Lavie, Chris Dyer, and Eduard Hovy. Humor recognition and humor anchor extraction. In Proceedings ofthe 2015 conference on empirical methods in natural language processing, pages 2367–2376, 2015.

[49] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.

[50] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.

[51] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. In International Conference on Machine Learning, pages 57116–57198. PMLR, 2024.

[52] Tianyi Zhang\*, Varsha Kishore\*, Felix Wu\*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SkeHuCVFDr.

[53] Yichi Zhang and Wen Zhang. Cause: Towards causal knowledge graph embedding. In China Conference on Knowledge Graph and Semantic Computing, pages 17–28. Springer, 2023.

[54] Yibo Zhao, Jiapeng Zhu, Can Xu, and Xiang Li. Cello: Causal evared and toxicity detection with meta-toxic knowledge graph. 2024.

[55] Yibo Zhao, Jiapeng Zhu, Can Xu, and Xiang Li. Enhancing llm-based hatred and toxicity detection with meta-toxic knowledge graph. arXiv preprint arXiv:2412.15268, 2024.

[56] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

```jsonl
In-Context CRG using GPT-4o
{
" entities ": {
" high_heels ": {
description ": "A
fashionable shoe with a
Rectified In-Context CRG
tall heel .",
effects ": [" elevates
appearance ", " causes foot discomfort "] { " entities ": {
}, " high_heels ": {
" feet ": { " properties ": [" A
" description ": " Human feet fashionable shoe with a
depicted before and after tall heel ."]
wearing high heels .", },
" effects ": [" get deformed " feet ": {
or stressed ", " cause " properties ": [" wears
pain "] high heel " , " bandages
}, applied at places "]
" fashion ": { },
description ": " Cultural or " fashion ": {
social drive to look " properties ": [" Cultural or
stylish or attractive .", social drive to look
effects ": [" pushes stylish or attractive ."]
people to wear high heels },
", " compromises comfort "] II discomfort ": {
■ " properties [" painful ",
} II
}, negative_physical_effects_on
" causal_relationships ": [ _feet "]
{" cause ": " fashion ", " effect ": }
{" cause ": " high_heels ", " effect " high_heels "} , }, " causal_relationships ": [
": " foot_discomfort "}, {" cause ": " fashion ", " effect ":
{ " cause ": " foot_discomfort ", " " feet wears high heels "},
effect ": " {" cause ": " feet wears high
negative_physical_effects_on heels ", " effect ": Ⅱ
_feet " } discomfort "}
] ]
} }
```  
Figure 3: Comparison of In-Context CRGs (Causal Reasoning Graphs): GPT-4o generated vs. Rectified version. The text spans with $6 6 \quad , 5 9$ on the left and $\ " \overrightarrow { \mathbf { \nabla } } \mathbf { \overrightarrow { \Gamma } }$ on the right in the GPT-4o generated CRG are rectified (removed/modified) to obtain the rectified CRG, where the spans with $\yen 9$ on the left and “ ” on the right are the resulting changes  
Fig. 3 compares in-context CRGs before and after rectification, highlighting the changes that took place.

![](images/b2ac9ef060a5de3a47ddd7630373f319fdc49870095dcf0c44ec42e77fcaa48e.jpg)  
Figure 4: Example for Humor (in this case, Satire) Understanding and Detection Tasks

Fig. 4 shows examples for Humor (in this case, Satire) Understanding and Detection Tasks, listing the input, task, and ground truth.

## C Task-specific Queries

Satirical Image Understanding. Why is this image funny/satirical?

Meme Caption Generation. Why is this meme funny?

Satirical Image Detection. You are an AI expert in detecting humor or satire. User gives you an image, and you have to make a choice "Y" or "N". Instructions: Users image has 2 halves called yes and but, and the combination of those might make no sense at all, or be extremely funny. Your job is to find out which one it is and output Y if its EXTREMELY funny and N for otherwise. Output format: one character, exactly either "Y" or "N"

Multimodal Sarcasm Detection. Is this image with the text funny/sarcastic? Give your final answer as ‘«YES»’ or ‘«NO»’

![](images/ec4a3e858645037be14b018c6f5b23cbbec724f4161c0d5c085f39e74298ca5b.jpg)  
Figure 5: Example Image from the YesBut Dataset

Corresponding to Fig. 5, the rationale and ground truth for CoT can be described as follows - Rationale.

Subject: Fashion footwear choices Premise: The image contrasts how high heels appear aesthetically versus their physical impact

Punchline: What looks beautiful and fashionable in design causes visible physical discomfort and strain

Irony: People willingly sacrifice comfort for style and appearance

Satirical commentary: The stark contrast between the elegant presentation of fashionable shoes and the resulting physical consequences they inflict on the wearer’s feet

Ground Truth Answer. The image is funny since it shows how wearing high heels in the name of fashion ends up causing a lot of physical discomfort to the user.

## E Ablation Analysis

<table><tr><td>No. of ICE</td><td>Ablation</td><td>ROUGE-L</td><td>BLEU</td><td>BERTScore</td><td>Avg. Score</td></tr><tr><td rowspan="2">0</td><td>WITH DEFN.</td><td>0.2131</td><td>0.0211</td><td>0.8718</td><td>0.3687</td></tr><tr><td>CARGO-T</td><td>0.2219</td><td>0.0245</td><td>0.8715</td><td>0.3726</td></tr><tr><td rowspan="3">2</td><td>WITH DEFN.</td><td>0.2406</td><td>0.0309</td><td>0.8816</td><td>0.3844</td></tr><tr><td>UNRECTIFIED</td><td>0.2492</td><td>0.0327</td><td>0.8495</td><td>0.3771</td></tr><tr><td>CARGO-T</td><td>0.2534</td><td>0.0339</td><td>0.886</td><td>0.3911</td></tr><tr><td rowspan="3">5</td><td>WITH DEFN.</td><td>0.2478</td><td>0.0308</td><td>0.8844</td><td>0.3877</td></tr><tr><td>UNRECTIFIED</td><td>0.2466</td><td>0.0310</td><td>0.8497</td><td>0.3758</td></tr><tr><td>CARGO-T</td><td>0.2513</td><td>0.0318</td><td>0.8872</td><td>0.3901</td></tr></table>

Table 8: Ablation Analysis of CARGO-T on GPT-4o. The best and second-best results for a particular number of in-context examples are highlighted in bold and italic respectively (ICE - In-Context Examples)

We compare CARGO-T with - (a) WITH DEFN. A detailed definition of the causal reasoning graph (mentioned in Section E.1 of Appendix) is added to the prompt of CARGO-T (b) UNRECTIFIED

CARGO-T. Causal Reasoning Graphs of in-context examples are not rectified manually in CARGO-T in few-shot setting Note that WITH DEFN. is carried out in zero-shot setting; all the ablations are carried out in in-context learning setting otherwise. The ablation analysis is carried out on GPT-4o in 0, 2, and 5-shot settings. We can infer from Table 8 that - (1) CARGO-T shows a consistent improvement on the lexical ROUGE-L, BLEU metrics, as well as the Avg. Score (2) In in-context learning setting, CARGO-T performs better than UNRECTIFIED across all metrics, showing the importance of manually rectifying the causal reasoning graphs in in-context examples (3) CARGO-T performs better than WITH DEFN. on 3/4 and 4/4 metrics for zero-shot and in-context learning settings respectively, suggesting that - using the definition of causal reasoning graph along with the task-specific query might be confusing for the VLM.

## E.1 Definition of Causal Reasoning Graph used in WITH DEFN.

1. Entities: There is a set of entities (listed in "entities") — each entity can have properties that are either descriptions of/adjectives/adverbs qualifying that entity or (non-causal) relations with other entities. These are listed under “properties” attribute of each entity. e.g entity: ANIMAL, "properties": ["Cats and Dogs", “pet of HUMAN”, “hides under FURNITURE”], note that HUMAN AND FURNITURE are other entities entity: FIREWORKS, "properties": [ "Bright and colorful explosions in sky", "burnt by HUMAN”] note that the (non—causal) relationships are bidirectional, e.g. FIRECRACKERS (burnt by) HUMAN, HUMAN (burns) FIRECRACKER are same relationships. This relationship is present in "properties" list of any ONE of these entities e.g ("burns FIRECRACKERS" belongs to HUMAN[“properties"]) OR ("burnt by HUM" belongs to FIRECRACKER[“properties"]), BUT NOT BOTH

2. CAUSAL relationships: listed under “causal\_relationships”. First, we define an EVENT. A collection of entities (along with their (non—causal) relationships) describes an EVENT which is typically of the form "X (optionally) does Y (optionally) with/for/to Z", (a single entity can also be an EVENT) — an EVENT is basically a macro node and a causal relation is defined between events. A causal relation is listed under "causal\_relationships" as a dictionary "cause": EVENT\_1, "effect’: EVENT\_2. Each event is expressed in natural language which tells what the collection of entities means, for instance, “X (optionally) does Y (optionally) with/for/to Z”. For example, "cause": “HUMAN burns FIRECRACKER S”, "effect’: “ANIMALS” are frightened

## F Further Analysis of CARGO-T’s Reasoning Component

Input: Texts $T _ { 1 } , T _ { 2 } ;$ smoothing parameter   
$\alpha > 0$   
Output: $K L ( T _ { 1 } | | T _ { 2 } )$   
Function   
PREPROCESSANDTOKENIZE(Text)   
return lowercase(Text) split on   
whitespace   
$/ /$ Tokenize both texts   
T ← PREPROCESSANDTOKENIZE $( T _ { 1 } )$   
$\mathcal { T } _ { 2 }$ ← PREPROCESSANDTOKENIZE $( T _ { 2 } ) ;$   
$/ /$ Build vocabulary   
$\mathcal { V }  \mathcal { T } _ { 1 } \cup \mathcal { T } _ { 2 } ;$   
$V  | \nu | ;$   
$/ /$ Count tokens   
foreach $i \in \{ 1 , 2 \}$ do   
$\forall w \in \mathcal { V } \colon \bar { c } _ { i } ( w ) \gets \# \{ t \in \mathcal { T } _ { i } : t = w \} \mathrm { : }$   
end   
$/ /$ Compute smoothed distributions   
for $i \in \{ 1 , 2 \}$ do   
$Z _ { i } \stackrel { \setminus } {  } | \bar { T _ { i } } | + \alpha V ;$   
$\forall w \in \mathcal { V } \colon \mathcal { D } _ { i } ( w ) \gets \frac { c _ { i } ( w ) + \alpha } { Z _ { i } } ;$   
end   
$\mathcal { P }  \mathcal { D } _ { 1 } , \mathcal { Q }  \mathcal { D } _ { 2 } ;$   
// Compute KL divergence   
$D \gets 0 ;$   
foreach $w \in \mathcal V$ do   
$D \gets D \mathrm { ~ + ~ } \mathcal { P } ( w )$ log $\frac { \mathcal { P } ( w ) } { \mathcal { Q } ( w ) }$   
end   
return $\smash { \frac { D } { \mathbf { \sigma } } }$   
Algorithm 1: KL-Divergence between token   
distributions

Input: Texts $T _ { 1 } , T _ { 2 } ;$ ; Upper Bound   
$U \in [ 0 , 1 ]$   
Output: $L S { \dot { F } } ( { \dot { T _ { 1 } } } | | T _ { 2 } ) \colon$ fraction of   
sentences in $\dot { T } _ { 1 }$ whose average   
similarity to $T _ { 2 }$ is below $U$   
Function PREPROCESSSENTENCES( T)   
return nltk.sent\_tokenize $( T )$   
Function EMBEDSENTENCES( S)   
return [ SentenceBert $( s ) \forall s \in S ]$   
$/ /$ Sentence-level preprocessing   
$S _ { 1 }$ ← PREPROCESSSENTENCES $( T _ { 1 } ) ;$   
$S _ { 2 } $ PREPROCESSSENTENCES $( T _ { 2 } ) ;$   
m $ \vert S _ { 1 } \vert , n  \vert S _ { 2 } \vert ;$   
$/ /$ Compute sentence embeddings   
$E _ { 1 }$ ← EMBEDSENTENCES $( S _ { 1 } ) ;$   
$E _ { 2 }$ ← EMBEDSENTENCES $( S _ { 2 } ) ;$   
$/ /$ Count low-similarity sentences   
count ← 0;   
foreach $e _ { 1 } \in E _ { 1 }$ do   
$/ /$ Cosine similarities to all   
sentences in $T _ { 2 }$   
simList ← [ cosineSim(e<sub>1</sub>, e<sub>2</sub>) ∀ $e _ { 2 } \in$   
$E _ { 2 } ] ;$   
$\textstyle { \bar { s } } \gets { \frac { 1 } { n } } \sum$ simList;   
$\mathbf { i f } \ \bar { s } < U$ then   
count ← count + 1;   
end   
end   
$/ /$ Compute and return the   
fraction   
return count   
m   
Algorithm 2: Low Similarity Fraction (LSF)