# Limitations of Automated Simulatability: LLM Simulators Can Bypass Explanations

Antonin Poché1,2, Fanny Jourdan1, Nils Feldhus³, Qianli Wang4, Jing Yang4,8, Simon Ostermann5,7,9, Nicholas Asher2,10, Philippe Muller2, Vera Schmitt4,6,7,9,

1IRT Saint Exupéry, 2IRIT, Université de Toulouse, 3University of Groningen, 4Technische Universität Berlin, 5Saarland University, 6Johannes Gutenberg-University Mainz, 'German Research Center for Artificial Intelligence (DFKI), 8BIFOLD – Berlin Institute for the Foundations of Learning and Data, 9Centre for European Research in Trusted AI (CERTAIN), 10CNRS, Correspondence: antonin.poche@irt-saintexupery.com

## Abstract

Simulatability is an evaluation protocol for explanations that quantifies their usefulness by how well they help a user predict a task model's outputs. Since human evaluation is costly, automated simulatability replaces human explainees with LLM simulators, as proposed in ConSim (Poché et al., 2025) for large-scale experiments. We qualitatively replicate and extend ConSim's ranking of explanation methods across the tested datasets, explanation families, and simulator LLMs, and identify two limitations. First, when class names are meaningful, simulators can obtain high simulatability by solving the classification task directly, without relying on the explanations. Second, class anonymization can reward explanations for leaking the hidden label mapping, a limitation we expose with a new classes-as-concepts baseline. These results are consistent with a shortcut hypothesis: in the tested settings, simulator predictions mainly rely on task priors, while explanations produce small changes. We derive recommendations for more robust automated simulatability evaluations.

## 1 Introduction

Explanations of a task model's predictions are often evaluated via faithfulness and complexity metrics (Jacovi and Goldberg, 2020), which only assess internal properties of the explanation, i.e., how well it reflects the task model's decision process or how concise it is, rather than whether the explanation is actually useful or plausible to an explainee.

Simulatability addresses this gap by evaluating explanations end-to-end: an explanation is useful insofar as it enables an explainee to predict the task model's behavior on held-out inputs (Kim et al., 2016). Early simulatability studies relied on humans in controlled user studies (Lage et al., 2019; Colin et al., 2022) which are costly and difficult to scale. To overcome this limitation, recent work substitutes human explainees with LLM simulators, yielding automated simulatability (Hase and Bansal, 2020; Poché et al., 2025).

![](images/d69a4da0c210cb90c63efb83242d5aaedb57b955af2ce8cf7e9754c6ecf0a09e.jpg)

![](images/0edbfd5ad4abc930ea6a1176213d311e643dc723a7600479879d8566061f6aed.jpg)

![](images/ac7b1341f6e6d765fb5cbab259e02f57d475a09aed58e46f660493c9908b5597.jpg)  
Figure 1: Overview of automated simulatability limitations. Automated simulatability explains how a task model works via ICL (in-context learning), and the LLM simulator must simulate its predictions on new samples. In theory, simulatability measures the usefulness of explanations. However, simulator predictions can remain unaffected by explanations, thus relying on task priors (Sec. 4). In anonymized settings, explanation scores can also reward hidden label recovery (Sec. 5).

We replicate and extend ConSim's experiments (Poché et al., 2025) to additional explanation types (attributions and rationales), LLM simulators, and datasets. As in the original paper, in nonanonymized experiments, we find that explanations produce small changes to simulator predictions. We hypothesize a shortcut: simulator predictions are dominated by task priors rather than explanation content. Echoing similar effects in ICL (incontext learning) (Liu et al., 2022; Jang et al., 2024) and, in human psychology, with goal neglect (Duncan et al., 1996). Then we present evidence consistent with this hypothesis. To prevent shortcuts, ConSim anonymizes class names; however, we show that this fix has its own limitations. We introduce a classes-as-concepts baseline, which trivially explains a prediction by naming the predicted class, and find that it outperforms every tested method under anonymization, showing that rankings can reward label leakage rather than meaningful explanations. We close with recommendations for robust simulatability protocols.

Our contributions are: (1) a qualitative replication of ConSim's protocol and extension to new datasets, explanation families (attributions and rationales), and LLM simulators; (2) the shortcut hypothesis and cross-family and cross-LLM evidence consistent with strong task-prior influence in nonanonymized settings; (3) the classes-as-concepts baseline, showing that anonymized evaluation can reward uninformative label leakage explanations; (4) concrete recommendations for more robust automated simulatability evaluation.

## 2 Background

Explanations types. Interpretability aims at understanding the behavior and decisions of task models. It can be divided into explanation types; we use three of them (thorough overview in App. A):

• Attributions are popular methods that quantify the contribution of input features to the prediction (Lundberg and Lee, 2017; Simonyan et al., 2014).

• Rationales are free-form textual justifications of a prediction (Camburu et al., 2018), most of the time LLM-generated (Turpin et al., 2023).

• Concept-based explanations map internal task model computations to human-understandable concepts, providing a higher-level view of what a task model has learned (Kim et al., 2018; Feldhus and Kopf, 2025). In this paper, we focus on posthoc unsupervised methods (Poeta et al., 2023).

Existing metrics largely assess the quality of all these methods with respect to task model behavior or generalization across inputs, rather than whether they are effectively communicated to an evaluator This gap motivates simulatability as a criterion for evaluating the effectiveness of explanations.

Simulatability refers to the degree to which explanations help a simulator to predict a task model's outputs (Kim et al., 2016; Lage et al., 2019). It shifts the evaluation focus from internal faithfulness to downstream utility: an explanation is useful insofar as it lets an evaluator reproduce the task model's behavior on held-out inputs. Early evaluations relied on controlled human user studies (Lage et al., 2019; Colin et al., 2022), which are difficult to scale. (Deeper related work in App. B).

Automated Simulatability. To overcome the scalability ceiling of human studies, recent work substitutes human explainees with LLM simulators (De Bona et al., 2024; Nguyen et al., 2024). However, the simulator's nature changes what is evaluated (Chan et al., 2022). In our paper, we replicate and extend ConSim (Poché et al., 2025), an automated simulatability framework for concept-based explanations. They notably introduce anonymized experiments (which we detail in Sec. 5) to force LLM simulators to rely on the explanations. They find consistent rankings across datasets, task models, and simulators.

## 3 Replication and extension of ConSim

Before analyzing the limitations, we verify that our implementation recovers the qualitative findings of ConSim (Poché et al., 2025). We then select a prompt format and concept interpretation for the analyses in Sections 4 and 5.

## 3.1 Replication setup

We detail ConSim experiments in App. C, but we invite the reader to refer to the original paper for the complete description (Poché et al., 2025). We call our replication old\_consim.

What is preserved: We keep the same simulatability logic, prompt-type families, and anonymized variants. We use the same methods (NoProjection is renamed neurons-as-concepts), and keep the TopK concept interpretation. We keep 3 of the four datasets: BIOS, IMDB, and Rotten Tomatoes. We also use deterministic sampling to balance the number of correct and incorrect task model predictions.

## What changes:

• Task models: We use one HuggingFace task model per dataset rather than retrain multiple architectures. Because their activations are not necessarily positive, we replace NMF with Semi-NMF (Ding et al., 2010).

• Evaluation stability: We use 50 seeds and split them into 10 disjoint groups of 5. Each grouped observation, therefore, aggregates 100 evaluation predictions rather than the 20 available from a single seed, reducing the discreteness and variance of the accuracy estimates.

• Coverage: We add AG News (Zhang et al., 2015) and GoEmotions (Demszky et al., 2020), replace Tweet Eval Emotion (Mohammad et al., 2018) by Emotion (Saravia et al., 2018), and treat BIOS slightly differently (App. D).

• Prompting: We reconstruct the original prompt setup as old\_consim and introduce new\_consim, which includes 3 changes: related information is interleaved with examples, importance values are verbalized, and evaluation samples are predicted one at a time. Details in App. E.

• Concept interpretation: We also test an LLMbased approach to concept interpretation, addressing one of ConSim's announced limitations.

## 3.2 Implementation and Reproducibility

Code, prompts, task model identifiers, seeds, outputs, and reproduction commands are available on GitHub. We rely on the library Interpreto (Poché et al., 2026) for attribution and conceptbased explanations. Interpreto concept-based explanations use the LanguageModel from NNsight (Fiotto-Kaufman et al., 2025) for model splitting and activation extraction, and overcomplete (Fel, 2025) for concept learning. The old\_consim implementation was available in Interpreto. Finally, plots rely on the code released with the original ConSim paper.

## 3.3 Reproduction and protocol selection

Qualitative reproduction. We first assess whether our reconstruction of the original old\_consim protocol recovers the qualitative findings reported by ConSim. Exact numerical replication is not expected because we use different task models, a different simulator, and a partially different set of datasets. We therefore evaluate whether the broad relative ordering of the explanation methods is preserved.

Figure 9 is a copy of figures reported in the original paper, while Fig. 10 shows our reconstruction. The main pattern is preserved: Vanilla SAE, ICA, and Semi-NMF rank above the no-explanation baseline, whereas PCA and SVD remain near the bottom. Some method-level positions differ, but we do not interpret these differences because the task models and decomposition setup have changed. Overall, our reproduction aligns with the original ConSim study.

Prompt-format selection. We next compare the reconstructed old\_consim format with our revised new\_consim format. The paired results are reported in Fig. 6 in App. F. new\_consim yields significantly higher simulatability scores for 8/10 prompt types. Nonetheless, the rankings of the two formats remain correlated (Pearson r = .87 and Spearman ρ = .87). Thus, new\_consim generally increases scores while broadly preserving the original behavior. We use new\_consim as the common prompt format in the following experiments. Note that on Qwen-3.5-9B, later conclusions hold with the old\_consim format and LLM-generated interpretations.

Concept-interpretation selection. Using new\_consim, we also compare TopK and LLMgenerated concept interpretations. TopK yields significantly higher scores for 5/6 applicable concept prompt types (Fig. 7 in App. F). We therefore use TopK as the default concept interpretation in the subsequent analyses.

Motivating observation. With both old\_consim and new\_consim, the non-anonymized score distributions overlap strongly across concept methods and their matched no-explanation baselines (Fig. 12). The next section explores this limitation.

## 4 Task-solving shortcuts in non-anonymized simulatability

The previous section showed that, when class names are visible, concept explanations produce only limited separation from their matched noexplanation baselines. We first test whether this behavior recurs across the tested explanation families and LLM simulators. We then examine the hypothesis that simulators primarily rely on their prior knowledge of the classification task, with only minor changes from the provided explanations.

<table><tr><td>Prompt element</td><td>B1</td><td>B2</td><td>C1</td><td>C2</td><td>C3</td><td>A</td><td>R</td></tr><tr><td>Task description</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Global explanations</td><td></td><td>一</td><td>X</td><td>X</td><td>X</td><td>一</td><td>一</td></tr><tr><td>Examples and predictions</td><td></td><td>X</td><td>1</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Local explanations</td><td>一</td><td>一</td><td>一</td><td>一</td><td>X</td><td>X</td><td>X</td></tr></table>

Table 1: Prompt types used for the cross-family shortcut experiments. B1 and B2 are no-explanation baselines. C1–C3 are concept prompts. A is the attribution prompt, where local explanations are word attributions for the learning examples. R is the rationale prompt, where local explanations are natural-language rationales for the learning examples. The evaluation phase never includes an explanation or a label.

## 4.1 Attributions and rationales extension

We extend the non-anonymized evaluation to concept explanations (C), feature attributions (A), and natural-language rationales (R) while keeping the simulator task and prompt structure fixed. All prompts include the task description and evaluation sample; they differ only in whether they contain global explanations, learning examples with task model predictions, and local explanations. Table 1 summarizes the resulting prompt types.

Following Sec. 3, concept prompts use new\_consim with TopK interpretations. We select one representative method per explanation family based on the Qwen-3.5-9B (Qwen-Team, 2026) rankings reported in App. I: Vanilla SAE for concepts, LIME for attributions, and Qwen3.5-2B for rationales. We first evaluate the complete grid with Qwen-3.5-9B, then repeat the representativemethod comparison with Llama-3.1-8B (Dubey et al., 2024), Gemma-4-12B (Abd et al., 2026), and Phi-4 (Abdin et al., 2024).

## 4.2 Consistent but small explanation gains across LLM simulators

With Qwen-3.5-9B, concept and attribution prompts are significantly above their matched noexplanation baselines (Fig. 2). However, the mean gains are small relative to the variability across experimental settings.

Gemma-4-12B and Phi-4 show similar rankings and conclusions (App. K): Explanations provide small gains over their matched baselines, with concept prompts showing the clearest improvements. This consistency is also visible at the prediction level: Qwen-3.5-9B, Gemma-4-12B, and Phi-4 agree on about 85% of their predictions (Fig. 19).

Three of four retained user-LLMs show similar small gains in explanation. Llama-3.1-8B reverses this pattern, demonstrating that automated simulatability remains simulator-dependent. Its lower agreement with the task model suggests that simulator capability may contribute to this difference.

![](images/859104f97e484a0b4ff0242f63c15e2b481e9115be4822e4a3fc60402011e13f.jpg)  
Figure 2: Qwen-3.5-9B prompt-type differences. Barplot comparing each selected prompt type to its matching no-explanation baseline.

## 4.3 Hypothesis

A small explanation can admit at least two interpretations. First, the tested explanations may provide little useful information beyond what the LLM simulator already knows. Second, the simulator may not use the explanations even when they contain relevant information, because directly predicting the gold label is easier than inferring the task model's behavior.

The consistency of the pattern across three explanation families and three LLM simulators makes a shared limitation of the evaluation protocol more plausible than independent failures of each explanation method, although the latter cannot be ruled out. We therefore investigate the following shortcut hypothesis: LLM simulator predictions are mainly determined by task priors, with explanations producing smaller changes.

## 4.4 Evidence for the shortcut hypothesis

High agreement without learning information. The B1 prompt provides the task description and class names but no learning examples, task model predictions, or explanations (Tab. 1). Nevertheless, B1 achieves simulatability scores above 0.7 across several datasets (penultimate violin in each group of Fig. 17). Because evaluation samples are balanced between correct and incorrect task model predictions, always predicting the gold label would yield an expected score of 0.5. A B1 score of 0.7, therefore, requires agreement with at least 40% of the task model's errors. Thus, high agreement can arise without learning examples, task model predictions, or explanations.

![](images/5397930b2acb2b6526a3595099caeb1628673c07d744652997fb69d7dd42b935.jpg)  
Figure 3: Prompt types mostly agree. Mean pairwise exact agreement between gold labels, task model predictions, and selected prompt-type predictions. Each cell is computed within a LLM simulator/dataset/class-subset triplet and averaged equally over all triplets.

Prompt types mostly agree. Fig. 3 reports exact agreement between predictions produced under the different prompt types. It shows strong agreement across prompt types, including the B1 (class-names-only) baseline. Therefore, in many cases, including examples or explanations does not change the simulator predictions. We observe even stronger agreement between prompts with learning examples; however, as shown in Fig. 17, this does not meaningfully change scores.

In addition, the dataset- and simulator-wise results in App. L show that predictions match the gold label on IMDB nearly 100% of the time across all prompt types, consistent with a strong reliance on prior knowledge.

Qualitatively different concepts have similar scores. The limited score separation is not solely due to equally uninformative concept descriptions. For the BIOS subset {dentist, physician, surgeon}, SVD's displayed global concept for dentist is “Scholarship, Ph.D., Professor, Faculty, .. ." and is opposed to the class, whereas Vanilla SAE displays the supportive descriptor “D.D.S., Dentistry, teeth, board-certified, ..." (Table 4). Yet, over C1–C3, their mean scores on this subset are equal (.712 each). More broadly, Fig. 4 shows strongly overlapping non-anonymized score distributions for all concept methods, despite such qualitative differences.

![](images/e8937a852a01c083ae52323adc3389d83e6fb7f2fb4a3a002f17d41ceb1ff2ba.jpg)  
Figure 4: Non-anonymized concept prompts overlap. Qwen-3.5-9B grouped-seed scores for TopK new\_consim C1-C3 prompts. The methods include qualitatively distinct concept decompositions, but their distributions largely overlap.

The harder the classification task, the clearer the differences. The clearest explanation gains occur on the selected GoEmotions subsets, which contain semantically confusable labels such as anger/annoyance/neutral and admiration/approval/caring. The task model's F1 score on this dataset is 0.541 (HuggingFace model id: SamLowe/roberta-base-go\_emotions). These subsets also have lower no-explanation performance, leaving more room for learning-phase information to help. This dataset dependence is consistent with the shortcut hypothesis: task priors dominate when the classification problem is easy, whereas explanation matters more when direct prediction is difficult.

Explicit simulatability framing does not improve agreement. Finally, we compare new\_consim with simulator\_consim, which explicitly instructs the LLM simulator to simulate the task model. All other prompt content and experimental settings are held fixed (App. E). Mentioning the task model generally reduces simulatability scores (Fig. 8). Thus, explicit simulator framing alone does not align with task model behavior.

These observations are consistent with the shortcut hypothesis: when direct classification is easy, simulator predictions change little when explanations are added. They do not establish that explanations are unused.

ConSim forces the simulator to rely on explanations by anonymizing the output classes. The next section examines whether anonymization successfully removes the shortcut.

## 5 Anonymization sensitivity to leakage

In theory, anonymization from ConSim (Poché et al., 2025) prevents LLM simulators from directly predicting meaningful class names. However, we show that explanations can reveal the hidden label mapping, creating a second shortcut.

## 5.1 Anonymized experiments

Anonymized experiments replace class names with generic labels such as Class\_1 and Class\_2. The simulator must predict these labels during evaluation. This removes the direct semantic link between the input and the expected output, preventing the simulator from solving the classification task solely using class names. Instead, it must infer the label mapping from the learning-phase examples and explanations. Without learning-phase information, the anonymized B1 baseline is expected to perform at the chance level.

## 5.2 Classes-as-concepts

Anonymization hides the output labels but not the semantic information contained in explanations. An explanation may therefore reveal which original class each anonymous label corresponds to.

To demonstrate this failure mode, we introduce the classes-as-concepts baseline, where each concept corresponds directly to one class. In a non-anonymized prompt, this produces redundant explanations, such as associating the prediction “surgeon" with the concept “surgeon." In an anonymized prompt, however, it associates Class\_0 with “surgeon," directly revealing the hidden label mapping.

Figure 5 shows that classes-as-concepts obtain the highest scores across all anonymized prompt types. The pairwise comparisons in Fig. 25 confirm that they outperform all other concept methods. This ranking does not indicate a better explanation of task model behavior: the baseline succeeds by revealing the mapping between anonymous labels and class names.

Anonymizing the explanation strings would not fully solve this issue, since synonyms or other semantic descriptions could reveal the same mapping.

Anonymized simulatability can therefore reward label leakage rather than explanation quality. We recommend including classes-as-concepts, or a similar label-leakage baseline, whenever anonymized prompts are used.

![](images/e8c434dc8f20d0e21d2f980c1cb49bb8e8814c74500fa75855a927f21e9eff69.jpg)  
Figure 5: Classes-as-concepts in anonymized concept prompts. Violin plot on Qwen-3.5-9B concept scores with TopK new\_consim and anonymized prompt types.

## 6 Conclusions and recommendations

Conclusions. In this paper, we introduced a prompting protocol and qualitatively replicated the ConSim ordering. As in the ConSim paper, we observe that non-anonymized explanations provide limited separation from no-explanation baselines.

We show that this pattern generalizes across concepts, attributions, and rationales and LLM simulators. Although explanation gains are statistically significant, they remain small relative to their variability. We propose the shortcut hypothesis to explain this result: simulator predictions are often dominated by direct task solving, with explanations producing only secondary changes. We present several observations consistent with this hypothesis.

ConSim uses class anonymization to prevent direct task solving. However, we find that anonymized evaluations can instead reward explanations that solely reveal the hidden label mapping. We demonstrate this failure mode with the classesas-concepts baseline, which outperforms the other concept methods under anonymization despite providing no information on the task model.

Both limitations show that automated simulatability scores can be misleading. These findings lead to the following recommendations:

## Recommendations:

• Over-sample the task model errors to emphasize model-specific behavior. It spreads scores and makes them more interpretable.

• Include a label leakage baseline, ensuring the ranking favors meaningful explanations (Sec. 5)

• Use multiple LLM simulators, report simulatorspecific results, and favor better LLMs.

• Use enough seeds for stable estimates, and report effect sizes and statistical significance.

• Use novel or semantically difficult tasks that cannot be solved easily from pretrained task knowledge, reducing shortcuts opportunities (Sec. 4).

## Limitations

Our findings use an LLM simulator with up to 15B parameters without reasoning. They might not generalize to larger, better, or closed-source LLMs.

We focused our analysis on use cases that LLM simulators have surely already encountered during training or that they have sufficient semantic knowledge to solve. In other words, we did not work around contamination (Balloccu et al., 2024).

## Ethical statement

Coding AI assistants were used throughout the project to help debug, help scale the experiments on a cluster (we used 590 H100 hours), improve visualizations, and detail appendices. Writing AI assistants were mainly used to simulate reviews to improve the general coherence and narrative flow. Everything was human-reviewed at least once.

## Ackowledgements

Our work has benefited from the AI Cluster AN-ITI and the research programs DEEL¹ and FOR². ANITI is funded by the France 2030 program under the Grant agreement n°ANR-23-IACL-0002. DEEL and FOR are integrative programs of the AI Cluster ANITI, designed and operated jointly with IRT Saint Exupéry, with the financial support from its industrial and academic partners and the France 2030 program under the Grant agreement n°ANR-10-AIRT-01.

This work was granted access to the HPC/AI resources of IDRIS under the allocation AD011017304 made by GENCI. Granting access to an H100 partition on the Jean Zay super-cluster.

## References

Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon, Mayank Chaturvedi, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, Clément Farabet, Johan Ferret, and 281 others. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero

Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, and 8 others. 2024. Phi-4 technical report. Preprint, arXiv:2412.08905.

Eldar D Abraham, Karel D'Oosterlinck, Amir Feder, Yair Gat, Atticus Geiger, Christopher Potts, Roi Reichart, and Zhengxuan Wu. 2022. Cebab: Estimating the causal effects of real-world concepts on nlp model behavior. Advances in Neural Information Processing Systems (NeurIPS).

Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain, Thomas Wiegand, Sebastian Lapuschkin, and Wojciech Samek. 2024. Attnlrp: attention-aware layer-wise relevance propagation for transformers. In Proceedings of the International Conference on Machine Learning (ICML).

Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. 2018. Sanity checks for saliency maps. In Advances in Neural Information Processing Systems (NeurIPS).

B Ans, J Hérault, and C Jutten. 1985. Architectures neuromimétiques adaptatives: Détection de primitives. Proceedings of Cognitiva.

Pepa Atanasova, Oana-Maria Camburu, Christina Lioma, Thomas Lukasiewicz, Jakob Grue Simonsen, and Isabelle Augenstein. 2023. Faithfulness tests for natural language explanations. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Sebastian Bach, Alexander Binder, Grégoire Montavon, Frederick Klauschen, Klaus-Robert Müller, and Wojciech Samek. 2015. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. Public Library of Science (PloS One).

Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondrej Dusek. 2024. Leak, cheat, repeat: Data contamination and evaluation malpractices in closedsource LLMs. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 67–93, St. Julian's, Malta. Association for Computational Linguistics.

Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa-Anke, and Leonardo Neves. 2020. Tweeteval: Unified benchmark and comparative evaluation for tweet classification. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

Usha Bhalla, Suraj Srinivas, Asma Ghandeharioun, and Himabindu Lakkaraju. 2024. Towards unifying interpretability and control: Evaluation via intervention. arXiv preprint arXiv:2411.04430.

Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan

Leike, Jeff Wu, and William Saunders. 2023. Language models can explain neurons in language models. OpenAI.

Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. e-snli: Natural language inference with natural language explanations. Advances in Neural Information Processing Systems (NeurIPS).

Aaron Chan, Shaoliang Nie, Liang Tan, Xiaochang Peng, Hamed Firooz, Maziar Sanjabi, and Xiang Ren. 2022. Frame: Evaluating rationale-label consistency metrics for free-text rationales. ArXiv e-print.

Yanda Chen, Ruiqi Zhong, Narutatsu Ri, Chen Zhao, He He, Jacob Steinhardt, Zhou Yu, and Kathleen McKeown. 2024. Do models explain themselves? counterfactual simulatability of natural language explanations. In Proceedings of the International Conference on Machine Learning (ICML).

Julien Colin, Thomas Fel, Rémi Cadène, and Thomas Serre. 2022. What i cannot predict, i do not understand: A human-centered evaluation framework for explainability methods. Advances in Neural Information Processing Systems (NeurIPS).

Arthur H Copeland. 1951. A reasonable social welfare function. Technical report, mimeo, 1951. University of Michigan.

Fahim Dalvi, Abdul Rafae Khan, Firoj Alam, Nadir Durrani, Jia Xu, and Hassan Sajjad. 2022. Discovering latent concepts learned in bert. In Proceedings of the International Conference on Learning Representations (ICLR).

Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In proceedings of the Conference on Fairness, Accountability, and Transparency.

Francesco Bombassei De Bona, Gabriele Dominici, Tim Miller, Marc Langheinrich, and Martin Gjoreski. 2024. Evaluating explanations through llms: Beyond traditional user studies. Workshop in Advances in Neural Information Processing Systems (NeurIPS).

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. GoEmotions: A dataset of fine-grained emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4040–4054, Online. Association for Computational Linguistics.

Chris H. Q. Ding, Tao Li, and Michael I. Jordan. 2010. Convex and semi-nonnegative matrix factorizations. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Pedro Domingos. 2015. The master algorithm: How the quest for the ultimate learning machine will remake our world. Basic Books.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. ArXiv e-print.

John Duncan, Hazel Emslie, Phyllis Williams, Roger Johnson, and Charles Freer. 1996. Intelligence and the frontal lobe: The organization of goal-directed behavior. Cognitive psychology, 30(3):257–303.

Carl Eckart and Gale Young. 1936. The approximation of one matrix by another of lower rank. Psychometrika.

Thomas Fel. 2025. Overcomplete: A visionbased sae toolbox. https://github.com/ KempnerInstitute/overcomplete.

Thomas Fel, Victor Boutin, Mazda Moayeri, Rémi Cadène, Louis Bethune, Mathieu Chalvidal, Thomas Serre, and 1 others. 2023. A holistic approach to unifying automatic concept extraction and concept importance estimation. In Advances in Neural Information Processing Systems (NeurIPS).

Thomas Fel, Remi Cadene, Mathieu Chalvidal, Matthieu Cord, David Vigouroux, and Thomas Serre. 2021. Look at the variance! efficient black-box explanations with sobol-based sensitivity analysis. In Advances in Neural Information Processing Systems (NeurIPS).

Nils Feldhus and Laura Kopf. 2025. Interpreting language models through concept descriptions: A survey. In Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP.

Jaden Fiotto-Kaufman, Alexander Loftus, Eric Todd, Jannik Brinkmann, Koyena Pal, Dmitrii Troitskii, Michael Ripa, Adam Belfki, Can Rager, Caden Juang, and 1 others. 2025. Nnsight and ndif: Democratizing access to open-weight foundation model internals. In Proceedings of the International Conference on Machine Learning (ICML).

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. 2022. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

David Gunning, Mark Stefik, Jaesik Choi, Tim Miller, Simone Stumpf, and Guang-Zhong Yang. 2019. XAI—explainable artificial intelligence. Science Robotics.

Peter Hase and Mohit Bansal. 2020. Evaluating explainable AI: Which algorithmic explanations help users predict model behavior? In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Sture Holm. 1979. A simple sequentially rejective multiple test procedure. Scandinavian journal of statistics, pages 65–70.

Pingjun Hong and Benjamin Roth. 2026a. Do LLM self-explanations help users predict model behavior? evaluating counterfactual simulatability with pragmatic perturbations. ArXiv e-print.

Pingjun Hong and Benjamin Roth. 2026b. Not all explanations simulate equally: Comparing verbalized feature attributions and self-generated rationales. ArXiv e-print.

Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. 2019. A benchmark for interpretability methods in deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS).

Harold Hotelling. 1992. Relations between two sets of variates. In Breakthroughs in statistics: methodology and distribution. Springer.

Jing Huang, Zhengxuan Wu, Christopher Potts, Mor Geva, and Atticus Geiger. 2024. Ravel: Evaluating interpretability methods on disentangling language model representations. ArXiv e-print.

Aapo Hyvärinen and Erkki Oja. 2000. Independent component analysis: algorithms and applications. Neural networks.

Alon Jacovi and Yoav Goldberg. 2020. Towards faithfully interpretable nlp systems: How should we define and evaluate faithfulness? In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Joonwon Jang, Sanghwan Jang, Wonbin Kweon, Minjin Jeon, and Hwanjo Yu. 2024. Rectifying demonstration shortcut in in-context learning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4294–4321, Mexico City, Mexico. Association for Computational Linguistics.

Been Kim, Rajiv Khanna, and Oluwasanmi O Koyejo. 2016. Examples are not enough, learn to criticize! Criticism for Interpretability. In Advances in Neural Information Processing Systems (NeurIPS).

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and 1 others. 2018. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In Proceedings of the International Conference on Machine Learning (ICML).

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in Neural Information Processing Systems (NeurIPS).

Isaac Lage, Emily Chen, Jeffrey He, Menaka Narayanan, Been Kim, Samuel J Gershman, and Finale Doshi-Velez. 2019. Human evaluation of models built for interpretability. In Proceedings of the 2019 ACM Conference on Fairness, Accountability, and Transparency.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, and 1 others. 2023. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702.

Daniel D Lee and H Sebastian Seung. 1999. Learning the parts of objects by non-negative matrix factorization. nature.

Marvin Limpijankit, Yanda Chen, Melanie Subbiah, Nicholas Deas, and Kathleen McKeown. 2025. Counterfactual simulatability of LLM explanations for generation tasks. In Proceedings of the 2025 International Natural Language Generation Conference.

Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics.

Scott Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (NIPS).

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Alireza Makhzani and Brendan Frey. 2013. K-sparse autoencoders. ArXiv e-print.

Edmund Mills, Shiye Su, Stuart Russell, and Scott Emmons. 2023. Almanacs: A simulatability benchmark for language model explainability. ArXiv e-print

Saif Mohammad, Felipe Bravo-Marquez, Mohammad Salameh, and Svetlana Kiritchenko. 2018. Semeval-2018 task 1: Affect in tweets. In Proceedings of the 12th international workshop on semantic evaluation, pages 1–17.

Sharan Narang, Colin Raffel, Katherine Lee, Adam Roberts, Noah Fiedel, and Karishma Malkan. 2020. Wt5?! training text-to-text models to explain their predictions. arXiv preprint arXiv:2004.14546.

Andrew Ng and 1 others. 2011. Sparse autoencoder. CS294A Lecture notes.

Giang Nguyen, Ivan Brugere, Shubham Sharma, Sanjay Kariyappa, Anh Totti Nguyen, and Freddy Lecue. 2024. Interpretable table question answering via plans of atomic table transformations. ArXiv e-print.

Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Karl Pearson. 1901. Liii. on lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science.

Antonin Poché, Lucas Hervier, and Mohamed-Chafik Bakkay. 2023. Natural example-based explainability: a survey. In World Conference on eXplainable Artificial Intelligence, pages 24–47. Springer.

Antonin Poché, Alon Jacovi, Agustin Martin Picard, Victor Boutin, and Fanny Jourdan. 2025. Consim: Measuring concept-based explanations' effectiveness with automated simulatability. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Antonin Poché, Thomas Mullor, Gabriele Sarti, Frédéric Boisnard, Corentin Friedrich, Charlotte Claye, François Hoofd, Raphael Bernas, Nicholas Asher, Céline Hudelot, and 1 others. 2026. Interpreto: An explainability library for transformers. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 1–13.

Eleonora Poeta, Gabriele Ciravegna, Eliana Pastor, Tania Cerquitelli, and Elena Baralis. 2023. Conceptbased explainable artificial intelligence: A survey. CoRR.

Qwen-Team. 2026. Qwen3.5: Accelerating productivity with native multimodal agents.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research (JMLR).

Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. 2019. Explain yourself! leveraging language models for commonsense reasoning. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the ACM International Conference on Knowledge Discovery and Data Mining (KDD).

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. In Advances in Neural Information Processing Systems (NIPS).

Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. 2018. CARER: Contextualized affect representations for emotion recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3687–3697, Brussels, Belgium. Association for Computational Linguistics.

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. 2017. Learning important features through propagating activation differences. In Proceedings of the International Conference on Machine Learning (ICML).

K Simonyan, A Vedaldi, and A Zisserman. 2014. Deep inside convolutional networks: visualising image classification models and saliency maps. In Proceedings of the International Conference on Learning Representations (ICLR).

Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. 2017. Smoothgrad: removing noise by adding noise. In Workshop on Visualization for Deep Learning.

Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. 2014. Striving for simplicity: The all convolutional net. In Workshop Proceedings of the International Conference on Learning Representations (ICLR).

Student. 1908. The probable error of a mean. Biometrika, pages 1–25.

Jingyi Sun, Qianli Wang, Pepa Atanasova, Nils Feldhus, and Isabelle Augenstein. 2026. Investigating the interplay between contextual and parametric chainof-thought faithfulness under optimization. Preprint, arXiv:2605.24960.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In Proceedings of the International Conference on Machine Learning (ICML).

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. 2023. Language models don't always say what they think: Unfaithful explanations in chain-ofthought prompting. Advances in Neural Information Processing Systems (NeurIPS).

Qianli Wang, Tatiana Anikina, Nils Feldhus, Simon Ostermann, Sebastian Möller, and Vera Schmitt. 2025.

Cross-refine: Improving natural language explanation generation by learning in tandem. In Proceedings of the 31st International Conference on Computational Linguistics, pages 1150–1167, Abu Dhabi, UAE. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems (NeurIPS).

Bingsheng Yao, Prithviraj Sen, Lucian Popa, James Hendler, and Dakuo Wang. 2023. Are human explanations always helpful? towards objective evaluation of human natural language explanations. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Chih-Kuan Yeh, Been Kim, Sercan Arik, Chun-Liang Li, Tomas Pfister, and Pradeep Ravikumar. 2020. On completeness-aware concept-based explanations in deep neural networks. Advances in neural information processing systems.

Mateo Espinosa Zarlenga, Pietro Barbiero, Zohreh Shams, Dmitry Kazhdan, Umang Bhatt, Adrian Weller, and Mateja Jamnik. 2023. Towards robust metrics for concept representation evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).

Matthew D Zeiler and Rob Fergus. 2014. Visualizing and understanding convolutional networks. In Proceedings of the IEEE European Conference on Computer Vision (ECCV).

Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems (NIPS).

## A Explanations types

Interpretability aims at understanding the behavior and decisions of task models. It can be divided into different explanation types (also called explanation families (Poché et al., 2023)):

Attributions are popular methods for quantifying the contribution of input features (pixels, tokens, etc.) to the prediction. Here we use two types of such methods, perturbation-based (Zeiler and Fergus, 2014; Ribeiro et al., 2016; Lundberg and Lee, 2017; Fel et al., 2021), (Lundberg and Lee, 2017), and gradient-based (Simonyan et al., 2014; Shrikumar et al., 2017; Smilkov et al., 2017; Sundararajan et al., 2017; Hooker et al., 2019). We do not include architecture-based (Springenberg et al., 2014; Bach et al., 2015; Selvaraju et al., 2017) or attention-based (Achtibat et al., 2024).

Rationales Another explanation type is naturallanguage rationales, which are free-form textual justifications of a prediction (Camburu et al., 2018; Rajani et al., 2019; Narang et al., 2020; Wang et al., 2025). They are commonly generated through selfrationalization or elicited as chains of thought (Kojima et al., 2022; Wei et al., 2022). They can seem plausible to humans, while not faithful to the task model (Turpin et al., 2023; Jacovi and Goldberg, 2020; Atanasova et al., 2023; Lanham et al., 2023; Turpin et al., 2023; Sun et al., 2026).

Concept-based explanations map internal task model computations to human-understandable concepts, providing a higher-level view of what a task model has learned (Kim et al., 2018; Feldhus and Kopf, 2025). In this paper, we focus on post-hoc unsupervised methods (Poeta et al., 2023), that explain an already trained task model without prior knowledge of its concepts. This paradigm follows a three-stage pipeline (Fel et al., 2023; Poché et al., 2025):

1. Concepts learning: Split the task model into two (in our case, between the encoder and the classification), then hidden representations (here [CLS] tokens) are factorized into basis directions using dictionary learning. (PCA, NMF, SAE...) (Fel et al., 2023).

2. Concepts' interpretation: abstract obtained directions are assigned a semantic e.g. via top-k activating tokens (Dalvi et al., 2022; Geva et al., 2022) or LLM-generated labels. (Bills et al., 2023).

3. Concepts' importance: Not all concepts are used in a given prediction or are important for a class. concepts' contributions toward the predictions (local) or classes (global) are weighted using gradient-input (Shrikumar et al., 2017) attribution methods on the concepts-to-output function.

Concepts evaluations A growing body of work investigates what makes concept spaces meaningful and how to evaluate them. Geva et al. (2022) shows that transformer feed-forward layers naturally promote discrete, vocabulary-level concepts, providing empirical grounding for the idea that neural networks learn concept-like representations. Completeness-aware approaches (Yeh et al., 2020) argue that concept-based explanations should cover the full space of relevant concepts rather than cherry-picking the most salient ones. CEBaB (Abraham et al., 2022) estimates the causal effects of real-world concepts on NLP task model behavior through intervention-based evaluation. Bhalla et al. (2024) propose a unifying framework for evaluating interpretability methods via intervention, while RAVEL (Huang et al., 2024) benchmarks concept discovery methods on their ability to disentangle language model representations. Zarlenga et al. (2023) propose robust metrics for evaluating concept representation, focusing on stability and faithfulness under perturbations.

## B Simulatability details

Simulatability refers to the degree to which an explainee can correctly predict a task model's outputs when provided with its explanations (Kim et al., 2016; Lage et al., 2019). It shifts the evaluation focus from internal faithfulness to downstream utility: an explanation is useful insofar as it lets an evaluator reproduce the task model's behavior on held-out inputs. The DARPA XAI program formalized this intuition, defining simulatability as the ability of a user to predict a task model's behavior from its explanation (Gunning et al., 2019). Early evaluations relied on controlled human user studies (Lage et al., 2019; Colin et al., 2022), which are difficult to scale. Hase and Bansal (2020) distinguishes between forward simulation (predicting a task model's output from an input and explanation) and counterfactual simulation (predicting behavior on a perturbed input after observing the original input, output, and explanation). Chen et al. (2024) apply counterfactual simulatability to self-generated natural language explanations, showing that models do not consistently explain themselves on perturbed inputs. Limpijankit et al. (2025) extend counterfactual simulatability to generation tasks, finding that explanations help more for skill-based than knowledge-based tasks. Hong and Roth (2026b) and Hong and Roth (2026a) compare verbalized feature attributions with self-generated rationales under counterfactual simulation, showing that CoT rationales provide substantially stronger simulation signals than attribution-based explanations. Yao et al. (2023) augment simulatability with a helpfulness signal measuring downstream gains at finetuning and inference, arguing that simulatability alone falls short for evaluating human-annotated explanations. Mills et al. (2023) introduce AL-MANACS, a benchmark for evaluating language model explainability through simulatability, covering multiple explanation types and tasks. To overcome the scalability ceiling of human studies, recent work substitutes human explainees with LLM simulators.

## C ConSim experiments details

In ConSim (Poché et al., 2025), the authors proposed an automated simulatability metric for concept-based explanations. The simulator is introduced to the classification task in an initial phase, sees samples and model predictions, and in a learning phase, and is then asked to predict the model's outputs on held-out samples. Their prompts vary in which elements are shown: two no-explanation baselines and three concept-explanation settings, with anonymized variants described in 5. They then aggregate scores through Copeland-style pairwise comparisons (Copeland, 1951), so methods are only compared on matched experimental settings.

Prompt types They define 5 prompt types, two baselines without explanations (B1 and B2), three concept-based prompts (C1, C2, and C3), and the anonymized versions of these prompts. Prompttypes overview in Tab. 1, details are given in App. E.1, and anonymized experiments are detailed in Sec. 5.

Datasets and models The original experiments cover BIOS10 (De-Arteaga et al., 2019), IMDB (Maas et al., 2011), Rotten Tomatoes (Pang and Lee, 2005), and Tweet Eval Emotion (Barbieri et al., 2020). They evaluate several model families: DistilBERT (Sanh et al., 2019), T5 (Raffel et al., 2020), and Llama-3-8B (Dubey et al., 2024), including positively fine-tuned DistilBERT and T5 variants required by non-negative matrix factorization (NMF).

Decomposition and interpretation methods The concept extraction methods are NMF (Lee and Seung, 1999), Sparse Auto-Encoders (SAE) (Ng et al., 2011; Makhzani and Frey, 2013), ICA (Hyvärinen and Oja, 2000), PCA (Pearson, 1901; Hotelling, 1992), SVD (Eckart and Young, 1936), and a NoProjection baseline where neurons are treated as concepts. Concepts are communicated either through TopK words (Dalvi et al., 2022; Geva et al., 2022) (called CMAW in the original paper and sometimes MaxAct in the literature) or through alignment to existing labels (o1CA). The reported ranking places NMF, SAE, and ICA above others, PCA and SVD are worse than the baselines, while TopK is more reliable than o1CA.

Samples and seeds Each prompt contains 40 selected samples per seed, split into 20 learningphase and 20 evaluation-phase samples. The selection is balanced so that half of the samples are correctly predicted by the task model and half are misclassified, making the task about simulating the task model rather than solving the underlying dataset. The main experiments use five random seeds, plus additional seeds for selecting the number of concepts.

## D Class-subset construction

BIOS (De-Arteaga et al., 2019) and GoEmotions (Demszky et al., 2020) each define 28 labels. Running every simulatability configuration on all 28 labels would too many samples to learn the task model behavior on each class. Furthermore, it would be much more complicated. We therefore run the simulatability evaluation on fixed threeclass subsets. Keeping the same cardinality is important as it allows to compare scores between class-subsets and aggregate them. Three classes provided a compromise between retaining confusable decisions and keeping the experiments tractable.

We selected the subsets from confusion matrices between the gold labels and the predictions of the corresponding off-the-shelf task models, Fannyjrd/roberta-bios-biased and SamLowe/roberta-base-go\_emotions. We prioritized groups with large off-diagonal confusion relative to their class frequency, while limiting overlap between groups. Table 3 reports the resulting subsets. The two BIOS subsets containing professor are the only overlap among the retained subsets within either dataset.

<table><tr><td>Property</td><td>old_consim</td><td>new_consim</td><td>simulator_consim</td></tr><tr><td>Task framing</td><td>Assign a class to each sample</td><td>Assign a class to the evalua- tion sample</td><td>Reproduce the class assigned by another classifier, even when disagreeing</td></tr><tr><td>Learning block</td><td>Texts, contributions, and pre- dictions in separate blocks</td><td>Text, label, and contributions interleaved per sample</td><td>Same as new, with “Model&#x27;s prediction&quot; replacing “Label”</td></tr><tr><td>Importance display</td><td>Concept IDs with -/-/+/++</td><td>Interpreted concept descrip- tors with verbal categories</td><td>Identical to new</td></tr><tr><td>Evaluation requests</td><td>One request containing 20 numbered samples</td><td>20 requests containing one sample each</td><td>20 requests containing one sample each</td></tr><tr><td>Requested output</td><td>20 lines of Sample_i: class</td><td>One class name</td><td>One classifier-predicted class name</td></tr></table>

Table 2: Controlled comparison of the three ConSim prompt specifications. New versus old changes prompt organization, importance rendering, and evaluation batching. Simulator versus new changes only task-facing wording and field names.

<table><tr><td>Dataset</td><td>Global class IDs</td><td>Class names</td></tr><tr><td>BIOS</td><td>[6, 19, 25]</td><td>dentist, physician, surgeon</td></tr><tr><td>BIOS</td><td>[21, 22, 26]</td><td>professor, psychologist, teacher</td></tr><tr><td>BIOS</td><td>[1, 21, 24]</td><td>architect, professor, software engineer</td></tr><tr><td>BIOS</td><td>[9, 11, 18]</td><td>filmmaker, journalist, photographer</td></tr><tr><td>GoEmotions</td><td>[2, 3,27]</td><td>anger, annoyance, neutral</td></tr><tr><td>GoEmotions</td><td>[0, 4, 5]</td><td>admiration, approval, caring</td></tr></table>

Table 3: Three-class subsets retained for BIOS and GoEmotions. IDs refer to the original datasets’global label indices.

The subsets constrain sample selection so that both the label and the task model prediction fall in the subset. For both datasets, concept decompositions are fitted to representations from all 28 classes, and their configured number of concepts is also computed from the full 28-class label space. Thus, a subset experiment evaluates how explanations from a global concept space convey the task model's behavior in a single confusable three-class decision problem.

For each subset and random seed, we cache 40 test samples, of which 20 are used as learningphase examples and 20 as evaluation samples. Selection is stratified so that, over the complete 40- sample set, half of the task model predictions are correct and half are errors; every class in the subset is represented among both pools whenever the data permit. This balancing makes evaluation accuracy measure agreement with the task model rather than ordinary accuracy against the gold labels. The split and final order are deterministic for a fixed dataset, subset, sample count, and seed. We use 50 seeds for every retained subset.

## E ConSim prompt formats

We evaluate three prompt specifications. old\_consim reconstructs the organization used in ConSim (Poché et al., 2025); new\_consim reorganizes the same information and requests one prediction at a time; and simulator\_consim keeps the new organization but explicitly frames the task as reproducing a task model. All three use the same cached task model predictions and the same 40 selected samples for a fixed dataset, class subset, and seed. Consequently, differences between specifications do not come from resampling.

## E.1 Prompt types and shared inputs

The experimental prompt types are listed in Table 1. Prefixing a prompt type with A gives its class-anonymized counterpart: AB1, AC1, AB2, AC2, and AC3. In those prompts, displayed class names and expected answers are replaced by Class\_i. The historical format numbers these labels by their order within the subset, whereas the new and simulator formats retain the datasets’ global class IDs. The upper-bound prompt supported by the historical implementation, which exposes local explanations for evaluation samples, is not included in our experiments.

For C1-C3, global explanations associate each displayed class with its most important concepts. C3 additionally associates each learning example with the concepts contributing to the task model's prediction. B1 and C1 contain no learning examples; B2, C2, and C3 contain the same 20 examples and task model predictions. Every type is evaluated on the remaining 20 samples.

## E.2 Old and new organization

Table 2 summarizes the structural differences. In old\_consim, the system message first lists all learning texts, then all local-contribution entries when present, and finally all task model predictions. Concept interpretations are listed separately, while global and local importances use concept IDs and the symbols -, -, +, and ++. Its single user message contains all 20 evaluation texts, numbered Sample\_20 through Sample\_39.

In new\_consim, each learning block instead interleaves the text, task model prediction, and optional local explanation. A concept ID and its interpretation are displayed together, and importance symbols are verbalized as Very opposed, Opposed, Supportive, or Highly supportive. The shared system message is paired with 20 separate user messages, each containing one evaluation text. This removes the need for the LLM simulator to maintain a 20-line output protocol and lets every evaluation prediction receive its own generation budget.

The following shortened C3 excerpts preserve the exact field organization while replacing sample text and explanation contents with placeholders:

Sample\_0: [learning text]   
Concepts contributions for Sample\_0: {1: '+', 3:   
'++′}   
Sample\_0: pos   
User: Sample\_20: [evaluation text]   
Sample\_21: [evaluation text]   
new\_consim.   
Sample\_0:   
Text: [learning text]   
Label: pos   
Concepts contributions: {C3 ([interpretation   
]): Highly supportive}   
User: Evaluation sample:   
Text: [one evaluation text]   
Label:

## E.3 Explicit simulator framing

simulator\_consim inherits sample handling, prompt types, concept rendering, and evaluation batching from new\_consim. Its instruction begins: “You are simulating a text classifier. For the given evaluation sample, predict the class label this classifier would assign."It then states: “Your goal is to reproduce the classifier's output, even when you would otherwise disagree." Learning and evaluation fields use Model's prediction instead of Label. No explanations, predictions, or class mappings are otherwise changed.

The simulator grid uses the six datasets, their retained class subsets, all seven concept configurations (Semi-NMF, ICA, PCA, SVD, Vanilla SAE, neurons-as-concepts, and classes-as-concepts), 50 seeds, and the ten standard and anonymized prompt types. It uses TopK interpretations for methods requiring an interpretation; classes-as-concepts directly uses class names and stores no interpretation key. The simulator grid therefore matches the TopK portions of the old and new grids; LLM-generated concept interpretations are not evaluated under simulator framing.

## E.4 Generation, parsing, and scoring

Every prompt group has 20 expected task model labels. New and simulator groups produce 20 independent generations, from which one allowed class label is parsed per generation. Old groups produce one generation expected to contain 20 indexed lines.

The old format costs less because all 20 samples are answered together; comparable new-format efficiency would require caching.

For every specification, unparsable outputs are treated as invalid rather than incorrect. A score is reported as the number correct divided by the number valid, only when at least 14 of the 20 expected labels are valid; otherwise, the prompt-group score is undefined. Accuracy is therefore conditional on valid responses, separating formatting failures from disagreement with the task model.

In practice, less than 1% of predictions were invalid, so we do not report it.

## F Protocol comparison

Figures 6–8 report paired differences between the prompt and interpretation variants considered in the main text. Scores are matched on all remaining experimental factors, including dataset, task model, class subset, explanation method, grouped seed, prompt type, and, when applicable, concept interpretation. Bars show mean paired score differences over five-seed-grouped observations, error bars show one standard deviation, and tick labels report Holm-corrected two-sided one-sample t-tests against zero. Bold tick labels indicate corrected p < .05.

![](images/15c7a161b955c4d6389c33fab0366e263572ba5b4b2f43609009521dc9ee0ba3.jpg)  
Figure 6: New- versus old-ConSim prompt format. Mean paired Qwen-3.5-9B score differences new\_consim - old\_consim. Positive bars favor the revised format.

The first comparison contrasts our revised new\_consim prompt format with the reconstructed old\_consim format. Positive values in Fig. 6 indicate higher simulatability under new\_consim. The revised format yields significantly higher scores for eight of the ten prompt types, significantly lower scores for C1, and no significant difference for B1. Despite these score shifts, the two formats remain strongly correlated across matched configurations, as reported in Sec. 3. We therefore use new\_consim in the subsequent experiments.

The second comparison contrasts LLMgenerated and TopK concept interpretations under new\_consim. Negative values in Fig. 7 indicate higher scores with TopK. This comparison is only meaningful for concept prompt types C1-C3 and AC1-AC3; baseline rows are included only because their scores are independent of the interpretation choice. TopK yields significantly higher scores for five of the six applicable prompt types, with no significant difference for AC1. We therefore use TopK as the default concept interpretation.

Thefinalcomparisoncontrasts simulator\_consim with new\_consim; the former differs only in explicitly instructing the LLM simulator to reproduce another classifier's predictions. Negative values in Fig. 8 indicate higher scores under new\_consim. We discuss this comparison as evidence about explicit simulator framing in Sec. 4.4.

![](images/69f676a512f0cc1d88d8b3827bb2374b1b780ba690a687efcbdcfb83ba86f2ab.jpg)

Figure 7: LLM-generated versus TopK concept interpretations. Mean paired Qwen-3.5-9B score differences LLM — TopK under new\_consim. Negative bars favor TopK. Only C1–C3 and AC1–AC3 represent applicable concept prompts.  
![](images/73b7e8fd3be955c0301aa63a62507cc3f8baed2b570bba850056cf5891d942c3.jpg)  
Figure 8: Explicit simulator framing versus new-ConSim. Mean paired Qwen-3.5-9B score differences simulator\_consim — new\_consim. Negative bars favor new\_consim.

Higher simulatability scores do not necessarily imply higher explanation quality. These comparisons are used to select a common experimental protocol and to study sensitivity to prompt framing, rather than to rank explanation methods.

## G ConSim reproduction plots with Qwen-3.5-9B judge

This appendix details the concept-method comparisons supporting Sec. 3. On the following methods: Semi-NMF, ICA, PCA, SVD, Vanilla SAE, neurons-as-concepts, and the no-explanation baseline. The pairwise reproduction matrices also exclude classes-as-concepts because it is introduced as a diagnostic baseline in Sec. 5, rather than as a method from the original ConSim comparison.

We average each five consecutive seeds into one of ten grouped-seed observations. For every method pair, we intersect rows on dataset, task model, class subset, grouped seed, and prompt type. A win-rate cell is the proportion of matched rows on which the row method has the higher score, with ties counting as half a win. Difference cells report the mean and standard deviation of the paired score differences. Bold values indicate paired Student t-tests (Student, 1908) that remain significant at α = .05 after Holm correction (Holm, 1979) across the 21 unique method pairs displayed in that matrix. Matrix order follows the same Copelandstyle (Copeland, 1951) count of pairwise win rates at least 50% as the original analysis .

## Copy of figures from the ConSim paper

To simplify comparison, we copy two figures from the ConSim paper. Fig. 9 is a screen capture from the original paper, from which we have the accord.

Reconstructed old ConSim. The old-format reproduction in Fig. 10 uses the four datasets closest to the original study: BIOS, Emotion, IMDB, and Rotten Tomatoes. It includes the five standard and five anonymized prompt types with TopK interpretations. Vanilla SAE ranks first, followed by ICA and Semi-NMF; PCA and SVD remain at the bottom. Seventeen of the 21 pairwise differences are significant after correction.

New ConSim. Figure 11 applies the same calculation to new\_consim and extends it to all six datasets. Every method pair has all 600 expected matched cells: ten dataset/class-subset settings, ten grouped seeds, and six aligned concept prompt types after matching each concept prompt to its corresponding baseline. Vanilla SAE remains first, followed by Semi-NMF and ICA. Differences among the lower-ranked methods are small despite their ordering. Holm correction retains 17 of the 20 pairwise differences that were significant before correction.

Score distributions. The matrices aggregate small paired differences that can be difficult to assess from rankings alone. Figure 12 therefore shows the underlying grouped-seed score distributions for the TopK grid, including classes-as-concepts as a diagnostic. Under non-anonymized C1–C3 prompts, method distributions overlap strongly with the matching no-explanation baselines in both formats. Most visible method separation occurs for anonymized AC1–AC3 prompts, where classesas-concepts can expose the hidden class mapping. This distinction motivates the separate anonymization analysis in Sec. 5 and App. M.

![](images/cc0d005df30bada870b4b4a07e9a7cd65fca626a7768d66469e540be666253de.jpg)

![](images/e6ada169a09c9ccb6d834c0dc27563db0f8991b28fc3f30947db4a4ffba25cec.jpg)  
Figure 9: Copy of original ConSim paper plots. Pairwise win rates (left) and mean score differences with standard deviations (right), using GPT-4o-mini scores on BIOS, Tweet Eval Emotion, IMDB, and Rotten Tomatoes.

<table><tr><td>Vanilla SAE</td><td>50%</td><td>67%</td><td>63%</td><td>67%</td><td>71%</td><td>73%</td><td>69%</td></tr><tr><td>ICA</td><td>33%</td><td>50%</td><td>51%</td><td>48%</td><td>56%</td><td>57%</td><td>59%</td></tr><tr><td>SemiNMF</td><td>37%</td><td>49%</td><td>50%</td><td>53%</td><td>57%</td><td>59%</td><td>57%</td></tr><tr><td>Neurons</td><td>33%</td><td>52%</td><td>47%</td><td>50%</td><td>54%</td><td>59%</td><td>59%</td></tr><tr><td>baseline</td><td>29%</td><td>44%</td><td>43%</td><td>46%</td><td>50%</td><td>53%</td><td>55%</td></tr><tr><td>PCA</td><td>27%</td><td>43%</td><td>41%</td><td>41%</td><td>47%</td><td>50%</td><td>51%</td></tr><tr><td>SVD</td><td>31%</td><td>41%</td><td>43%</td><td>41%</td><td>45%</td><td>49%</td><td>50%</td></tr><tr><td>Vanilla SAE</td><td></td><td>ICA SemiNMF</td><td>Neurons</td><td>baseline</td><td></td><td>PCA</td><td>SVD</td></tr></table>

<table><tr><td colspan="1" rowspan="9">Vanilla SAEVanillaICASemiNMFNeuronsbaselinePCASVD</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">0.05±0.10</td><td colspan="1" rowspan="1">0.03±0.10</td><td colspan="1" rowspan="1">0.05±0.10</td><td colspan="1" rowspan="1">0.05±0.10</td><td colspan="1" rowspan="1">0.07±0.10</td><td colspan="1" rowspan="1">0.07±0.10</td></tr><tr><td colspan="1" rowspan="1">-0.05±0.10</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">-0.02±0.10</td><td colspan="1" rowspan="1">-0.01±0.07</td><td colspan="1" rowspan="1">0.00±0.07</td><td colspan="1" rowspan="1">0.02±0.08</td><td colspan="1" rowspan="1">0.02±0.09</td></tr><tr><td colspan="1" rowspan="1">-0.03±0.10</td><td colspan="1" rowspan="1">0.02±0.10</td><td colspan="1" rowspan="1">一</td><td colspan="1" rowspan="1">0.01±0.08</td><td colspan="1" rowspan="1">0.02±0.07</td><td colspan="1" rowspan="1">0.03±0.10</td><td colspan="1" rowspan="1">0.04±0.10</td></tr><tr><td colspan="1" rowspan="1">-0.05±0.10</td><td colspan="1" rowspan="1">0.01±0.07</td><td colspan="1" rowspan="1">-0.01±0.08</td><td colspan="1" rowspan="1">一</td><td colspan="1" rowspan="1">0.00±0.06</td><td colspan="1" rowspan="1">0.02±0.08</td><td colspan="1" rowspan="1">0.02±0.09</td></tr><tr><td colspan="1" rowspan="1">-0.05±0.10</td><td colspan="1" rowspan="1">-0.00±0.07</td><td colspan="1" rowspan="1">-0.02±0.07</td><td colspan="1" rowspan="1">-0.00±0.06</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">0.01±0.08</td><td colspan="1" rowspan="1">0.02±0.08</td></tr><tr><td colspan="1" rowspan="2">-0.07±0.10</td><td colspan="1" rowspan="1">-0.02</td><td colspan="1" rowspan="1">-0.03</td><td colspan="1" rowspan="1">-0.02</td><td colspan="1" rowspan="2">-0.01±0.08</td><td colspan="1" rowspan="2">一</td><td colspan="1" rowspan="2">0.01±0.06</td></tr><tr><td colspan="1" rowspan="1">±0.08</td><td colspan="1" rowspan="1">±0.10</td><td colspan="1" rowspan="1">±0.08</td></tr><tr><td colspan="1" rowspan="1">-0.07±0.10</td><td colspan="1" rowspan="1">-0.02±0.09</td><td colspan="1" rowspan="1">-0.04±0.10</td><td colspan="2" rowspan="1">-0.02 -0.02±0.09±0.08</td><td colspan="2" rowspan="1">-0.01±0.06  一</td></tr><tr><td colspan="7" rowspan="1">ICA                PCASVDSemiNMFNeuronsbaselineSAE</td></tr><tr><td>Vanilla SAE</td><td>50%</td><td>67%</td><td>68%</td><td>71%</td><td>71%</td><td>69%</td><td>70%</td></tr><tr><td>SemiNMF</td><td>33%</td><td>50%</td><td>52%</td><td>56%</td><td>58%</td><td>56%</td><td>58%</td></tr><tr><td>ICA</td><td>32%</td><td>48%</td><td>50%</td><td>55%</td><td>58%</td><td>56%</td><td>59%</td></tr><tr><td>PCA</td><td>29%</td><td>44%</td><td>45%</td><td>50%</td><td>54%</td><td>51%</td><td>55%</td></tr><tr><td>SVD</td><td>29%</td><td>42%</td><td>42%</td><td>46%</td><td>50%</td><td>50%</td><td>55%</td></tr><tr><td>Neurons</td><td>31%</td><td>44%</td><td>44%</td><td>49%</td><td>50%</td><td>50%</td><td>54%</td></tr><tr><td>baseline</td><td>30%</td><td>42%</td><td>41%</td><td>45%</td><td>45%</td><td>46%</td><td>50%</td></tr><tr><td>Vanilla SAE</td><td>SemiNMF</td><td></td><td>ICA</td><td>PCA</td><td>SVD Neurons</td><td>baseline</td><td></td></tr></table>

Figure 10: Reconstructed old-ConSim concept-method comparison. Pairwise win rates (left) and mean score differences with standard deviations (right), using Qwen-3.5-9B scores on BIOS, Emotion, IMDB, and Rotten Tomatoes.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.05±0.11</td><td rowspan=1 colspan=2>0.05±0.11</td><td rowspan=1 colspan=1>0.06±0.10</td><td rowspan=1 colspan=3>0.06±0.11</td><td rowspan=1 colspan=1>0.06±0.11</td><td rowspan=1 colspan=1>0.07±0.11</td></tr><tr><td rowspan=1 colspan=1>-0.05±0.11</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.01±0.07</td><td rowspan=1 colspan=1>0.01±0.06</td><td rowspan=1 colspan=3>0.01±0.06</td><td rowspan=1 colspan=1>0.02±0.08</td><td rowspan=1 colspan=1>0.02±0.08</td></tr><tr><td rowspan=2 colspan=1>-0.05±0.11</td><td rowspan=2 colspan=1>-0.01±0.07</td><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>0.00</td><td rowspan=2 colspan=3>0.01±0.05</td><td rowspan=2 colspan=1>0.01±0.04</td><td rowspan=2 colspan=1>0.01±0.04</td></tr><tr><td rowspan=1 colspan=1>±0.04</td></tr><tr><td rowspan=2 colspan=1>-0.06±0.10</td><td rowspan=1 colspan=1>-0.01</td><td rowspan=1 colspan=2>-0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=2 colspan=1>0.01±0.04</td></tr><tr><td rowspan=1 colspan=1>±0.06</td><td rowspan=1 colspan=2>±0.04</td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>±0.04</td></tr><tr><td rowspan=2 colspan=1>-0.06±0.11</td><td rowspan=2 colspan=1>-0.01±0.06</td><td rowspan=1 colspan=2>-0.01</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=3></td><td rowspan=2 colspan=1>0.00±0.05</td><td rowspan=2 colspan=1>0.01±0.05</td></tr><tr><td rowspan=1 colspan=2>±0.05</td><td rowspan=1 colspan=1>±0.04</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>-0.06</td><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=2>-0.01</td><td rowspan=3 colspan=6>0.00±0.03-0.01 -0.01-0.01 -0.00±0.03  一±0.04±0.04±0.05</td></tr><tr><td rowspan=1 colspan=1>±0.11</td><td rowspan=1 colspan=1>±0.08</td><td rowspan=1 colspan=2>±0.04</td><td rowspan=1 colspan=1>±0.04</td></tr><tr><td rowspan=1 colspan=1>-0.07±0.11</td><td rowspan=1 colspan=1>-0.02±0.08</td><td></td><td></td></tr></table>

Figure 11: New-ConSim concept-method comparison. Pairwise win rates (left) and mean score differences with standard deviations (right), using Qwen-3.5-9B scores on all six datasets.

![](images/91d33caa12c85ed4c3e1a73de80ec91dd4a7cded60f75f290221dfe450023c7f.jpg)

![](images/8bf4a9917fc80b8dca2c0a4f614b954b844c1da406c42a21030398a8bdf2eeb6.jpg)  
Figure 12: Old- and new-ConSim score distributions. Qwen-3.5-9B grouped-seed scores for TopK concept configurations under old (top) and new (bottom) prompting. Non-anonymized C1–C3 distributions largely overlap across methods and the matched no-explanation baseline; anonymized AC1–AC3 prompts produce greater separation.

![](images/7f110a8226830efb0f7a0048c2c3c0bf291be5684a872d76aea5c600441fdec5.jpg)  
Figure 13: Dataset-wise old/new prompt-format differences. Mean paired Qwen-3.5-9B score differences new\_consim — old\_consim, with one-standard-deviation error bars. Bars aggregate methods, TopK and LLM interpretations, class subsets, and grouped seeds within each dataset and prompt type.

## H Old- and new-ConSim comparison with Qwen-3.5-9B

This appendix disaggregates the aggregate old/new comparison in Fig. 6 by dataset. We pair old\_consim and new\_consim scores on every remaining index level: dataset, task model, class subset, grouped seed, method, interpretation, and prompt type. The plotted value is always new\_consim — old\_consim, so positive bars favor the new organization. The comparison spans 9,160 grouped seeds exact matches.

Figure 13 shows substantial dataset heterogeneity. After Holm correction across all 60 displayed dataset-prompt-type tests, 33 differences are significant: 26 favor the new format and seven favor the old format. Emotion and GoEmotions each have six significantly positive prompt types. IMDB has no significantly positive difference and a significantly negative AC1 difference, while Rotten Tomatoes combines four positive and three negative differences. Thus, reorganizing the prompt usually helps, especially for anonymized concept prompts, but it does not induce a uniform offset across datasets or prompt types.

![](images/c829c62d4e2824d71b63631dc4fc394a33b62d1781cef6a10652264477ba5bce.jpg)

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.01±0.04</td><td rowspan=1 colspan=1>0.01±0.04</td><td rowspan=1 colspan=2>0.02±0.04</td><td rowspan=1 colspan=3>0.02±0.04</td><td rowspan=1 colspan=1>0.02±0.04</td></tr><tr><td rowspan=1 colspan=1>-0.01±0.04</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>0.00±0.02</td><td rowspan=1 colspan=2>0.00±0.03</td><td rowspan=1 colspan=3>0.00±0.02</td><td rowspan=1 colspan=1>0.01±0.03</td></tr><tr><td rowspan=2 colspan=1>-0.01±0.04</td><td rowspan=2 colspan=1>-0.00±0.02</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>0.00</td><td rowspan=2 colspan=3>0.00±0.02</td><td rowspan=2 colspan=1>0.00±0.03</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=7 colspan=1>-0.02±0.04-0.02±0.04</td><td rowspan=3 colspan=1>-0.00±0.03</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>-0.00</td><td rowspan=3 colspan=1>0.00±0.03</td></tr><tr><td rowspan=2 colspan=1>±0.03</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=4 colspan=1>-0.00±0.02</td><td rowspan=3 colspan=1>-0.00</td><td rowspan=3 colspan=2>0.00</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=2></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=2 colspan=1>0.00±0.03</td></tr><tr><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=2>±0.03</td></tr><tr><td rowspan=1 colspan=1>-0.02±0.04</td><td rowspan=1 colspan=1>-0.01±0.03</td><td rowspan=1 colspan=1>-0.00±0.03</td><td rowspan=1 colspan=6>-0.00    -0.00±0.03    ±0.03</td></tr></table>

Figure 14: Concept-method ranking. Pairwise win rates (left) and mean score differences with standard deviations (right) for TopK C1–C3 prompts. Every pair contains 300 matched cells.

## I Explanation-method rankings with Qwen-3.5-9B

We rank candidate methods separately within each explanation family using Qwen-3.5-9B scores. All comparisons are non-anonymized: concept methods use new\_consim, TopK interpretations, and C1-C3. Baselines and classes-as-concepts are not ranking candidates. For each method pair, we intersect all remaining experimental keys, count a tie as half a win, and order methods by the same Copeland-style non-loss count used in the concept analysis. Difference matrices report paired means and standard deviations; bold cells survive Holm correction within that explanation family.

Concepts. The 6 compared decompositions are PCA (Pearson, 1901; Hotelling, 1992), SVD (Eckart and Young, 1936), (Ans et al., 1985; Hyvärinen and Oja, 2000), Semi-NMF (Ding et al., 2010), and SAEs (Ng et al., 2011; Makhzani and Frey, 2013; Domingos, 2015).

Figure 14 compares six methods on 300 matched cells per pair. Vanilla SAE ranks first, with win rates of 60–66% against every contender and mean advantages of 0.013–0.018. All five of these differences remain significant after Holm correction over the 15 concept-method pairs. Six pairs are significant overall, the additional pair being ICA over SVD. We therefore retain Vanilla SAE as the representative concept method.

<table><tr><td>Class</td><td>Method</td><td>Displayed global descriptor</td></tr><tr><td>dentist</td><td>SVD</td><td>Scholarship, scholarship, Ph.D., Professor, Faculty</td></tr><tr><td>dentist</td><td>Vanilla SAE</td><td>(opposed) D.D.S., Dentistry, teeth, board- certified, registry (supportive)</td></tr><tr><td>physician</td><td>SVD</td><td>Scholarship, scholarship, Ph.D., Professor, Faculty</td></tr><tr><td>physician</td><td>Vanilla SAE</td><td>(supportive) Hematology/Oncology, hospi- tal/clinic, Hematology, patient- centered, preventative (support- ive)</td></tr><tr><td>surgeon</td><td>SVD Vanilla</td><td>None displayed</td></tr><tr><td>surgeon</td><td>SAE</td><td>medicine/transfusion-free, hos- pital/clinic, medical-surgical, Dr, therapeutic (supportive)</td></tr></table>

Table 4: Qualitative BIOS TopK concept examples. Representative words or phrases from the first displayed globally important concept for each class in the {dentist, physician, surgeon} subset. Descriptors and directions are reproduced from the non-anonymized new\_consim prompts.

Attributions. The 10 compared attribution methods are: Saliency (Simonyan et al., 2014), Occlusion (Zeiler and Fergus, 2014), Lime (Ribeiro et al., 2016), KernelShap (Lundberg and Lee, 2017), SmoothGrad (Smilkov et al., 2017), Integrated Gradients (Sundararajan et al., 2017), SquareGrad (Adebayo et al., 2018), VarGrad (Hooker et al., 2019), and Sobol (Fel et al., 2021). Note that we use the gradient-input (Shrikumar et al., 2017) version of all gradient-based methods.

<table><tr><td>lime</td><td>50%</td><td>56%</td><td>55%</td><td>56%</td><td>57%</td><td>59%</td><td>55%</td><td>60%</td><td>62%</td><td>61%</td></tr><tr><td>square_grad</td><td>44%</td><td>50%</td><td>51%</td><td>51%</td><td>53%</td><td>55%</td><td>52%</td><td>55%</td><td>55%</td><td>56%</td></tr><tr><td>kernel_shap</td><td>45%</td><td>49%</td><td>50%</td><td>54%</td><td>54%</td><td>53%</td><td>52%</td><td>58%</td><td>55%</td><td>56%</td></tr><tr><td>occlusion</td><td>44%</td><td>49%</td><td>46%</td><td>50%</td><td>52%</td><td>55%</td><td>51%</td><td>54%</td><td>54%</td><td>57%</td></tr><tr><td>Int. gradients</td><td>43%</td><td>47%</td><td>46%</td><td>48%</td><td>50%</td><td>52%</td><td>52%</td><td>50%</td><td>54%</td><td>52%</td></tr><tr><td>gradient_shap</td><td>41%</td><td>46%</td><td>47%</td><td>46%</td><td>48%</td><td>50%</td><td>50%</td><td>50%</td><td>54%</td><td>55%</td></tr><tr><td>var_grad</td><td>45%</td><td>48%</td><td>48%</td><td>49%</td><td>48%</td><td>50%</td><td>50%</td><td>52%</td><td>52%</td><td>50%</td></tr><tr><td>smooth_grad</td><td>40%</td><td>45%</td><td>42%</td><td>46%</td><td>50%</td><td>50%</td><td>48%</td><td>50%</td><td>49%</td><td>52%</td></tr><tr><td>sobol</td><td>38%</td><td>46%</td><td>45%</td><td>46%</td><td>46%</td><td>46%</td><td>48%</td><td>51%</td><td>50%</td><td>50%</td></tr><tr><td>saliency</td><td>39%</td><td>44%</td><td>44%</td><td>42%</td><td>48%</td><td>46%</td><td>50%</td><td>48%</td><td>50%</td><td>50%</td></tr><tr><td></td><td>lime square_grad</td><td>kernel_shap</td><td>occlusion</td><td>Int. gradients</td><td>gradient_shap</td><td></td><td>var_grad smooth_grad</td><td></td><td>sobol</td><td>saliency</td></tr></table>

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00±0.01</td><td rowspan=1 colspan=1>0.00±0.01</td><td rowspan=1 colspan=1>0.00±0.01</td><td rowspan=1 colspan=1>0.00±0.01</td><td rowspan=1 colspan=1>0.00±0.02</td><td rowspan=1 colspan=1>0.00±0.01</td><td rowspan=1 colspan=1>0.00±0.02</td><td rowspan=1 colspan=1>0.00±0.01</td><td rowspan=1 colspan=1>0.00±0.01</td></tr><tr><td rowspan=2 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.01</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td></tr><tr><td rowspan=2 colspan=1>-0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.02</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.02</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.02</td></tr><tr><td rowspan=2 colspan=1>-0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.01</td></tr><tr><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.02</td></tr><tr><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=2 colspan=1>0.00±0.02</td><td rowspan=2 colspan=1>0.00±0.01</td></tr><tr><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td></tr><tr><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.00</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.02</td><td rowspan=2 colspan=1>0.00±0.01</td></tr><tr><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.01</td></tr><tr><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.01</td></tr><tr><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>-0.00±0.01</td><td rowspan=2 colspan=1>0.00±0.01</td></tr><tr><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td></tr><tr><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>0.00±0.02</td></tr><tr><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1>±0.01</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1>-0.00±0.02</td><td rowspan=1 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1>-0.00±0.01</td><td rowspan=1 colspan=1>-0.00±0.02</td><td rowspan=1 colspan=1></td></tr></table>

Figure 15: Attribution-method ranking. Pairwise win rates (left) and mean score differences with standard deviations (right) for A1 prompts. Pairs contain 100 matched cells, except kernel-SHAP pairs, which contain 90. No comparison is significant after Holm correction.

Figure 15 compares ten methods. LIME ranks first and wins 55–62% of its matched comparisons, but its mean advantages are only 0.001–0.004. None of the 45 attribution-method differences remains significant after Holm correction. LIME is therefore a deterministic first-ranked representative, not evidence of a reliably superior attribution method.

Rationales. Figure 16 compares the two available rationale generators on 100 matched cells. Qwen3.5-2B has a 57% win rate over Llama-3.2- 3B-Instruct, but the mean difference is only 0.0017 and is not significant $( p = . 3 8 )$ . We use Qwen3.5- 2B as the representative while treating the two generators as statistically indistinguishable under this metric. We note that a Qwen model judged the other Qwen model, so the comparison is not fully fair. Nonetheless, due to the score difference, we do not expect it to impact the conclusions in any way.  
![](images/775886b5d445fd67c34952dc4ccf98ef258035b753b9838546e0b5308feb5af5.jpg)

![](images/ace51ed4f6a5b51054f48450f2255322d3deab9eda7a9548b4ed0be32139cf66.jpg)  
Figure 16: Rationale-generator ranking. Pairwise win rates (left) and mean score differences with standard deviations (right) for R1 prompts. The single comparison contains 100 matched cells and is not significant.

## J Explanation-family score distributions

Figure 17 shows the grouped-seed score distributions for the representatives selected in App. I: Vanilla SAE concepts (C1–C3), LIME attributions (A), and Qwen3.5-2B rationales (R). B1 is the matched baseline for C1, while B2 is the matched baseline for C2, C3, A, and R. Baseline copies are first averaged within each explanation-family specification and then averaged equally across concepts, rationales, and attributions. This prevents repeated concept metadata configurations from receiving extra weight.

Each displayed series contains 100 grouped cells: ten grouped seeds for each of ten dataset/classsubset settings. These pooled distributions are not dataset-balanced because the four BIOS subsets contribute 40 cells and the two GoEmotions subsets contribute 20, while each other dataset contributes ten. All grouped cells are present.

Qwen, Phi, and Gemma show broad overlap, with pooled prompt-type means concentrated around 0.64–0.68. Their concept improvements over matching baselines are generally only one to three score points and vary substantially by dataset. Rationales do not reliably improve over B2 for any of these three judges, and attributions improve significantly only for Qwen. Llama behaves differently: B2 has the highest pooled mean, while A, C2, C3, and R are lower. Thus, no explanation family has a stable advantage across judges and datasets; App. L quantifies these paired differences.

Llama being the least performant model of the bunch, we could dismiss its results as a failure of the judge. With this, results are much more aligned.

llama3.1-8b  
gemma-4-12b  
![](images/a9e4dfa35a1b55e2319c031decf48d9f1d8c29484833341f226492823ee50ebd.jpg)

qwen3.5-9b  
![](images/62ed6d77fa234b7e41ebfefe82951a0029686d381e18bb2c01695c2f29bf0b9c.jpg)

![](images/f82d8d36af4ccd8795befc7b67eb218c9cd678d14e7e40867d53a695a6ed5532.jpg)

phi4  
![](images/78510ee8753c917d1984d4633056318e96e0dae42541bf3b7718f9ffc5aca4ab.jpg)  
Figure 17: Selected explanation-family score distributions. Grouped-seed scores by dataset for Llama-3.1-8B, Qwen-3.5-9B, Gemma-4-12B, and Phi-4 judges. A is LIME, C1–C3 are TopK Vanilla SAE prompts, and R uses Qwen-3.5-2B rationales; B1 and B2 are the matching no-explanation baselines. Horizontal black lines mark the corresponding baseline means within each dataset

![](images/c1e6c0a19137a42c85cadaf1bc6c3fc004da89827a76a98a14b1569db9362c24.jpg)  
(a) Qwen-3.5-9B

![](images/32032b495f63813bbe6834f9d65334eba4c852f7ff0627cc5fd79f036c6b51f3.jpg)  
(b) Gemma-4-12B

![](images/085440a948d0ad3a436f81007f38a3dfd33986938a3555673b5ac6e9ca27f574.jpg)  
(c) Phi-4

![](images/6f12bdf0f68cc6fbb3778fdc99472cdd2ec94a9a2896ec11f757df51af43c506.jpg)  
(d) Llama-3.1-8B  
Figure 18: Matched explanation-baseline differences across LLM simulators. Bars report mean paired simulatability-score differences, with one-standard-deviation error bars. C1 is compared with B1; C2, C3, attribution (A), and rationale (R) are compared with B2. Positive values indicate higher scores when explanations are included. Displayed p-values are obtained from paired Student t-tests and Holm-corrected.

## K Matched explanation-baseline differences across LLM simulators

Figure 17 compares each explanation prompt with its matched no-explanation baseline: C1 is compared with B1, while C2, C3, A, and R are compared with B2. Comparisons are matched on dataset, task model, class subset, and grouped seed, yielding 100 paired grouped-seed observations for each contrast and LLM simulator. Bars report the mean paired simulatability-score difference, with one standard deviation. The displayed p-values come from two-sided paired Student t-tests with Holm correction across the five comparisons for each LLM simulator. Subset settings are weighted equally, so BIOS and GoEmotions receive greater aggregate weight than single-setting datasets.

Qwen-3.5-9B. C1, C2, C3, and A significantly outperform their matched baselines. R is slightly below B2, but the difference is not significant.

Gemma-4-12B. Gemma-4-12B shows the same ordering as Qwen-3.5-9B. C1 and C3 significantly outperform their baselines, while the smaller differences for C2, A, and R are not significant.

Phi-4. All five differences are positive. C1, C2, and C3 are significant, while A and R are not.

Llama-3.1-8B. All five differences are negative. C2, C3, A, and R are significantly below B2, while C1 is not significantly different from B1.

Cross-LLM simulator pattern. Qwen-3.5-9B, Gemma-4-12B, and Phi-4 show a similar ordering: concept prompts provide the largest gains, while attribution and rationale provide smaller or null gains. Llama-3.1-8B is the main exception.

These results support using multiple LLM simulators and reporting effect sizes rather than relying only on statistical significance.

![](images/fb46ccfb5c492fd1ca9554278bac38ea97df8a7172bf6893acbb098a87e0ef66.jpg)

![](images/5a52f5471ef71825025715913fec430960f141c5ecf83aba0b928c3a87e93f41.jpg)  
Figure 19: Aggregate prediction exact agreement. Mean pairwise exact agreement among prompt types, gold labels, and task model predictions (left), and among LLM simulator after stacking prompt types (right). Agreements are averaged with equal weight; missing predictions are handled pairwise.

## L Prompt-prediction and judge exact agreement

This appendix analyzes the sample-level class predictions. We retain the non-anonymized representatives selected in App. I: Vanilla SAE with TopK interpretations for C1–C3, LIME for A, and Qwen3.5-2B for R, together with B1 and B2. To avoid stitching together disagreeing duplicate baseline generations, B1 and B2 use the attribution specification as one declared baseline source for every judge. The analysis covers all six datasets, including GoEmotions and BIOS.

Exact agreement is the percentage of pairwisecomplete predictions assigned the same class label. It is computed separately within each setting and then averaged with equal weight. The overall prompt matrix uses 40 judge/dataset/class-subset settings.

Figure 19 summarizes the two views. Offdiagonal prompt-type agreements range from 77% to 90%. This includes the B1 baseline, which has neither learning examples nor explanations, showing that much of the LLM simulator prediction remains unchanged across prompt types.

Figures 20–23 disaggregate the prompt matrix by judge and dataset. Agreement varies substantially by dataset and judge. On AG News and BIOS, Qwen-3.5-9B, Gemma-4-12B, and Phi-4 generally retain moderate-to-high agreement with the task model, whereas Llama-3.1-8B is more variable. On IMDB, prompt/task model agreement is only 45– 53% for every judge, close to the task model's 50% agreement with gold in this deliberately balanced correct/error sample. Agreement with gold is much larger: B1 and C1 exceed 90% for every judge. The simulators mostly follow the sentiment labels, providing direct evidence of task solving instead of task model simulation in that setting; the diverse LLM simulators agree on this pattern.

These matrices are descriptive, not significance tests, and high prompt-to-prompt exact agreement does not by itself prove that explanations are unused. Combined with the byte-identical baselines, small explanation-baseline score differences, and the IMDB failure mode, it nevertheless supports the shortcut hypothesis: much of the simulator prediction is determined by the text classification task and judge prior rather than explanation-specific information.

![](images/4573dbfc298f3c9b1fad5f0e2feb4e044f9563687f9ead286ea3a83ab97ce195.jpg)

![](images/d7f2a3f0ab04822d7b096f0c373e630f7404c1637869e33fce457be12f467bc8.jpg)

E
<table><tr><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>53%</td><td rowspan=1 colspan=5>46%48%50% 50%50% 4</td><td rowspan=1 colspan=2>7% 47%</td></tr><tr><td rowspan=1 colspan=1>53%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>42%</td><td rowspan=1 colspan=1>44%</td><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=2>49%50%</td><td rowspan=1 colspan=1>47%</td><td rowspan=1 colspan=1>43%</td></tr><tr><td rowspan=1 colspan=1>46%</td><td rowspan=1 colspan=1>42%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>75%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=2>63%63%</td><td rowspan=1 colspan=1>69%</td><td rowspan=1 colspan=1>78%</td></tr><tr><td rowspan=1 colspan=1>48%</td><td rowspan=1 colspan=1>44%</td><td rowspan=1 colspan=1>75%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>71%</td><td rowspan=1 colspan=1>76%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>85%</td><td rowspan=1 colspan=1>87%</td></tr><tr><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>71%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>79%</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>72%</td></tr><tr><td rowspan=2 colspan=1>50%50%</td><td rowspan=1 colspan=1>49%</td><td rowspan=1 colspan=1>63%</td><td rowspan=1 colspan=1>76%</td><td rowspan=1 colspan=1>79%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>87%</td><td rowspan=1 colspan=1>75%</td><td rowspan=1 colspan=1>72%</td></tr><tr><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=1>63%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>87%1</td><td rowspan=1 colspan=1>00%</td><td rowspan=1 colspan=1>73%</td><td rowspan=1 colspan=1>70%</td></tr><tr><td rowspan=2 colspan=1>47%47%</td><td rowspan=1 colspan=1>47%</td><td rowspan=1 colspan=1>69%</td><td rowspan=1 colspan=1>85%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>75%</td><td rowspan=1 colspan=1>73%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>81%</td></tr><tr><td rowspan=1 colspan=1>43%</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>87%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>81%1</td><td rowspan=1 colspan=1>00%</td></tr></table>

GE
<table><tr><td colspan="11">OL</td></tr><tr><td></td><td>100%</td><td>49%</td><td>50%</td><td>52%</td><td>51%</td><td>54%</td><td>54%</td><td>54% 53%</td><td></td></tr><tr><td></td><td>49%</td><td>100%</td><td>58%</td><td>67%</td><td>68%</td><td>71%</td><td>74%</td><td>69%</td><td>67% 70%</td></tr><tr><td>52%</td><td>50%</td><td>58%</td><td>100%</td><td>71%</td><td>68%</td><td>67%</td><td>64%</td><td>68%</td><td></td></tr><tr><td>51%</td><td></td><td>67%</td><td>71%</td><td>100%</td><td>69%</td><td>86%</td><td>83%</td><td>88% 88%</td><td></td></tr><tr><td>54%</td><td></td><td>71%</td><td>68% 68% 67%</td><td>69% 86%</td><td>100% 75%</td><td>75% 75%</td><td></td><td>70% 68%</td><td>83%</td></tr><tr><td>54%</td><td></td><td>74%</td><td>64%</td><td>83%</td><td>75%</td><td>100%</td><td>92%</td><td>86%</td><td>92% 100% 86% 82%</td></tr><tr><td>54%</td><td></td><td>69%</td><td>68%</td><td>88%</td><td>70%</td><td>86%</td><td>86%</td><td></td><td>100%88%</td></tr><tr><td>53%</td><td></td><td>67% 70%</td><td></td><td>88%</td><td>68%</td><td></td><td></td><td></td><td>88% 100%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>83% 82%</td><td></td><td></td><td></td></tr></table>

IMDB  
![](images/07a54e157749394651ce0bb3a681bb3659d8ed454312ad5f0258daa0d0c1fa88.jpg)

![](images/cf67f65ef43ef5259d17709282b653c1e70e9b049ab3b6b6ab001f02270d62ae.jpg)  
Figure 20: Qwen-3.5-9B prediction exact agreement by dataset. Exact agreement among gold labels, task model predictions, and selected prompt types, averaged equally over class subsets within each dataset.

![](images/174ddfbfe87036934227ab33f58df765c3d227b0530ecccddd5bf6b90cd50d42.jpg)

![](images/2c8ed8e0eea6a6e537880d4eb778bd513f5e4447e8ad7410d14c0f79a41e0403.jpg)

E  
![](images/65ab0ecd32fa201e36a21ad49a46e42b63b02b7345ca5e882405bac9cf795f83.jpg)

GE
<table><tr><td colspan="8">GE</td></tr><tr><td>100%</td><td>49%</td><td>56%</td><td>57%</td><td>56%</td><td>57%</td><td>57%</td><td>58%</td><td>57%</td></tr><tr><td>49%</td><td>100%</td><td>60%</td><td>66%</td><td>67%</td><td>70%</td><td>72%</td><td>68%</td><td>65%</td></tr><tr><td>56%</td><td>60%</td><td>100%</td><td>81%</td><td>78%</td><td>75%</td><td>74%</td><td>81%</td><td>84%</td></tr><tr><td>57%</td><td>66%</td><td>81%</td><td>100%</td><td>81%</td><td>86%</td><td>84%</td><td>90%</td><td>92%</td></tr><tr><td>56%</td><td>67%</td><td>78%</td><td>81%</td><td></td><td>100% 88%</td><td>88%</td><td>83% 82%</td><td></td></tr><tr><td>57%</td><td>70%</td><td>75%</td><td>86%</td><td>88%</td><td>100%</td><td>93%</td><td>87%</td><td>84%</td></tr><tr><td>57%</td><td></td><td>72% 74%</td><td>84%</td><td>88%</td><td>93%</td><td>100% 86%</td><td></td><td>82%</td></tr><tr><td>58%</td><td>68%</td><td>81%</td><td>90%</td><td>83%</td><td>87%</td><td>86%</td><td>100% 90%</td><td></td></tr><tr><td>57%</td><td>65%</td><td>84%</td><td>92%</td><td>82%</td><td>84%</td><td>82%</td><td>90% 100%</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

IMDB  
![](images/7f0f20407472fbce795e31b3a48a106614d6fccd3307222eb1d26b06534e4ddf.jpg)

![](images/15f83dbf6254fb479a8b6c6708a355e0ddaa6a4b0bbeea39f31b6cda73c76557.jpg)  
Figure 21: Gemma-4-12B prediction exact agreement by dataset. Exact agreement among gold labels, task model predictions, and selected prompt types, averaged equally over class subsets within each dataset.

![](images/74e7ce5685e11b473be845696a36ea2062d2319712b9c3429400624b6006b2ab.jpg)  
E

![](images/5ddb6dc9564c571d503599a631b287990b562f62800897a3cee7bdb66af1df8a.jpg)

![](images/8c52815a3896cbaa128b0cad5f0c50ab4770599a96ecc43f9b11e20812a41172.jpg)

![](images/b906b960eac492c031a42176eb8f382d574bc752c145d5eb71f149377534f574.jpg)

![](images/b1f975bc5ce2c98f511eac1b7c5a8d2840f06ba44adadc078a6e437391f5d0ac.jpg)

![](images/ae4ee892a226eb5493bf10a37f27314b09b57ca5cd3caeceedeceb4490dd46e7.jpg)  
Figure 22: Llama-3.1-8B prediction exact agreement by dataset. Exact agreement among gold labels, task model predictions, and selected prompt types, averaged equally over class subsets within each dataset.

![](images/9b4016bd9a70d787d18f177948ec64387dfa92fd333435b870207373939a3f97.jpg)

![](images/aaa7f5ca7d6820836a178beeba5e419964558eb550bdacc3a16ab1d590775c79.jpg)

E  
![](images/fa4b77a934fc0d35fa9f1a465e46eb0b0b9e0c7d249205b7d959fc8b78943d28.jpg)

GE  
![](images/784453f99550d39d848e8f0fadc7ab8906251e3a01ecd042443b4a69849e1312.jpg)

IMDB  
![](images/245a4829e78696684d2b03a32474ae4999aaf8de1dd6d6bd5020bff4a3f3aa37.jpg)

RT  
![](images/6b1b37d94e8cf753e6be1467531efbc6dc61b2c4718ff9a73039ebb66f22ca45.jpg)  
Figure 23: Phi-4 prediction exact agreement by dataset. Exact agreement among gold labels, task model predictions, and selected prompt types, averaged equally over class subsets within each dataset.

## M Anonymized and non-anonymized new-ConSim comparisons

This appendix isolates the effect of class anonymization on the Qwen-3.5-9B conceptmethod ranking. Both matrices use new\_consim, TopK interpretations, all six datasets, and the paper grid of Semi-NMF, ICA, PCA, SVD, Vanilla SAE neurons-as-concepts, classes-as-concepts, and the matched no-explanation baseline. As in App. G, exact duplicates are averaged and every five seeds form one grouped-seed observation. Each method pair has 300 matched cells: ten dataset/class-subset settings, ten grouped seeds, and three prompt types. Holm correction is applied separately across the 28 method pairs in each matrix.

Non-anonymized prompts. Figure 24 aggregates C1–C3. Vanilla SAE ranks first, but the mean differences are small: every displayed gap rounds to at most 0.02. 10 of 28 pairwise differences are significant after correction. classes-as-concepts sits near the lower middle of the ranking and differs from the no-explanation baseline by only 0.002 on average, which is not significant. Thus, its direct use of class names as concept descriptions provides no material advantage when the prompt already reveals them.

Anonymized prompts. Figure 25 instead aggregates AC1-AC3. Classes-as-concepts now ranks first and beats every other contender after correction. Its mean advantage ranges from 0.022 over Vanilla SAE to 0.138 over the no-explanation baseline, and its pairwise win rates range from 55% to 88% against the retained contenders. Overall, 24 of 28 method pairs are significant. The explanation instead acts as a key for translating hidden Class\_i identifiers back to class names. The reversal therefore demonstrates that anonymized simulatability can reward label leakage rather than explanation quality.

<table><tr><td>Vanilla SAE</td><td>50%</td><td>61%</td><td>60% 66%</td><td>62%</td><td>62%</td><td>66%</td><td>63%</td></tr><tr><td>ICA</td><td>39%</td><td>50%</td><td>52%</td><td>53% 55%</td><td>57%</td><td>59%</td><td>57%</td></tr><tr><td>Neurons</td><td>40%</td><td>48%</td><td>50%</td><td>51%</td><td>53% 56%</td><td>57%</td><td>58%</td></tr><tr><td>SemiNMF</td><td>34%</td><td>47%</td><td>49%</td><td>50%</td><td>52% 54%</td><td>56%</td><td>54%</td></tr><tr><td>PCA</td><td>38%</td><td>45%</td><td>47%</td><td>48%</td><td>50%</td><td>51% 55%</td><td>52%</td></tr><tr><td>Classes</td><td>38%</td><td>43%</td><td>44%</td><td>46%</td><td>49% 50%</td><td>54%</td><td>54%</td></tr><tr><td>SVD</td><td>34%</td><td>41%</td><td>43%</td><td>44%</td><td>45% 46%</td><td>50%</td><td>51%</td></tr><tr><td>No expl.</td><td>37%</td><td>43%</td><td>42%</td><td>46%</td><td>48%</td><td>46% 49%</td><td>50%</td></tr><tr><td>Vanilla SAE</td><td></td><td>ICA Neurons</td><td>SemiNMF</td><td>PCA</td><td>Classes</td><td>SVD No expl.</td><td></td></tr></table>

<table><tr><td rowspan=3 colspan=1>Vanilla SAE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2>0.01</td><td rowspan=2 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>:0.04</td><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=1>±0.05</td></tr><tr><td rowspan=13 colspan=1>ICANeuronsSemiNMFPCAClassesSVDNo expl.</td><td rowspan=1 colspan=1>-0.01</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.03±</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>±0.02 ±</td><td rowspan=1 colspan=1>0.03 ±</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>-0.01</td><td rowspan=1 colspan=2>-0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=2>±0.02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.03±</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.03±</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=2>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=2>±0.03</td><td rowspan=1 colspan=1>±0.03</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.03</td><td rowspan=1 colspan=1>±0.03±</td><td rowspan=1 colspan=1>0.03 ±</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=2>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=2>±0.02</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.03</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.03</td><td rowspan=1 colspan=1>±0.03</td></tr><tr><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=2>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>±0.04</td><td rowspan=1 colspan=2>±0.02</td><td rowspan=1 colspan=1>±0.02</td><td rowspan=1 colspan=1>±0.03±</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>±0.03 ±</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=2>-0.01</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=1 colspan=1>-0.00</td><td rowspan=2 colspan=1>-0.00±0.03</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>0.00±0.04</td></tr><tr><td rowspan=1 colspan=1>±0.04±</td><td rowspan=1 colspan=2>0.03</td><td rowspan=1 colspan=1>±0.03</td><td rowspan=1 colspan=1>±0.03 ±</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>-0.02±0.05</td><td rowspan=1 colspan=2>-0.01±0.03</td><td rowspan=1 colspan=6>-0.00-0.00-0.00-0.00-0.00±0.03±0.04 ±0.03±0.03±0.04</td></tr><tr><td rowspan=1 colspan=1>Vanilla</td><td rowspan=1 colspan=9>ICA        PCA   SVDNeuronsSemiNMFClassesNo expl.SAE</td></tr></table>

Figure 24: Non-anonymized new-ConSim concept comparison. Pairwise win rates (left) and mean score differences with standard deviations (right) for C1–C3. Every pair contains 300 matched cells.

![](images/28d5dbb76661d87d7bbe5be311461b6917d816ee5576060a52caca82dc52db68.jpg)

![](images/5c3b1cb461e988d26778ccd88fa5cae091d7077046d8a683364ae9d1e3e7d5fa.jpg)  
Figure 25: Anonymized new-ConSim concept comparison. Pairwise win rates (left) and mean score differences with standard deviations (right) for AC1-AC3. Every pair contains 300 matched cells. Classes-as-concepts ranks first because its concept descriptions reveal the hidden class mapping.