# Every Part Has Its Role: Part-of-Speech Effects in LLM Post-Training

Laya Iyer Stanford University laya@stanford.edu

Chen Shani Tel Aviv University cshani@tau.ac.il

## Abstract

Large language models are typically trained with next-token prediction (NTP), which rewards predicting specific tokens rather than explicitly accounting for the semantic similarities between different words or expressions. Prior work has addressed this limitation by augmenting training data with minimally varying sentences that replace a target word with conceptually related alternatives, e.g., “The tired coach ran home” → {“The tired trainer ran home”, “The tired instructor ran home”, ...}. Such concept augmentation improves performance, but it remains unclear whether its effectiveness de pends on where in the sentence the concept is expressed. We reproduce and extend prior work beyond nouns to verbs and adjectives, asking whether the part of speech (POS) carrying the concept signal affects perplexity and downstream performance. We find different patterns across evaluation settings: concept augmentation improves perplexity most consistently on news, independent of the POS used, whereas verbs provide the most reliable downstream gains. Despite these behavioral differences, representational analyses using CKA and linear probing show that the conditions yielding the largest downstream gains are not necessar ily those that most alter the model’s internal representations. Together, these results suggest that the effectiveness of concept augmentation depends on where and how conceptual infor mation is introduced during training.

## 1 Introduction

Large language models (LLMs) are predominantly trained with next-token prediction (NTP), which rewards predicting the observed token in a sequence. While highly effective, NTP operates over individual tokens: different words are treated as distinct targets even when they express closely related meanings. Humans, by contrast, reason not only over words but also over concepts: semantic units that connect different linguistic expressions under shared meaning (Shani et al., 2023). For example, “trainer,” “coach,” and “instructor” are distinct lexical forms but closely related expressions of the same concept.

Recent work has begun to incorporate such conceptual relationships into language-model training. In particular, Iyer et al. (2026) introduce conceptaware post-training using synonym and hypernym supervision and show that concept signal can affect both language-model and downstream performance. However, their data-augmentation experiments focus on nouns, leaving open a basic question: does the linguistic category through which a concept is expressed affect the value of concept supervision?

Nouns, verbs, and adjectives encode different aspects of meaning: entities and categories, actions and relations, and properties and states, respectively. For example, the sentence “The tired coach ran home” can be varied by replacing the noun “coach” with “trainer,” the verb “ran” with “hurried,” or the adjective “tired” with “exhausted.” Each substitution introduces a conceptually related alternative while changing a different component of the sentence. Concept-aware training may therefore induce different forms of invariance depending on which part of speech carries the concept signal.

We extend the data-augmentation approach of Iyer et al. (2026) from nouns to verbs, adjectives, and a combination of all three POS. We retain the standard NTP objective, isolating the effect of where conceptual variation is introduced rather than changing the training loss itself. We post-train Llama-3-8B with synonym and hypernym augmentation across arXiv, news, and YouTube data, and evaluate the resulting models using held-out perplexity and seven downstream NLP tasks.

Our results show that POS meaningfully shapes the effects of concept augmentation. Perplexity effects depend strongly on post-training domain, while verb augmentation provides the most consistent downstream gains across experimental settings. However, the strongest POS varies across domains and tasks, and augmenting all three POSs together does not consistently outperform targeted augmentation. These findings suggest that conceptaware training should not treat all content words as interchangeable sources of concept signal: POS provides an additional design axis for concept supervision.

![](images/a5e25a31f3e3ccf3c8d06f58b2d094ac9fd551fb77c949921d61b6eab1c734f3.jpg)  
Figure 1: Illustration of POS-specific concept augmentation. For the same input sentence, we generate synonym and hypernym substitutions using either context-aware or dictionary-based concept extraction. Rows show augmentation applied to nouns, verbs, adjectives, or all three POS categories together.

## 2 Related Work

Limitations of next-token prediction. Several lines of work have questioned whether next-token prediction alone provides the right supervision for learning generalizable semantic structure. Bachmann and Nagarajan (2025) show that teacherforced NTP can fit the training distribution without learning the underlying solution required for generalization. At the semantic level, Shani et al. (2023) argue that token-level probabilities can fragment probability mass across different word choices of the same concept, while Zhang et al. (2026) show that standard NTP treats semantically valid alternative continuations as competing targets. Our work does not replace NTP, but asks whether introducing structured semantic variation into the training data changes what is learned under the same objective.

Concept-aware language modeling. Recent work has introduced concept-level structure at several points in the language-modeling pipeline. Shani et al. (2023) study conceptual structure in

LLMs and motivate moving beyond purely tokenlevel representations. At the architectural level, team et al. (2024) introduce Large Concept Models, which autoregressively predict sentence-level semantic embeddings rather than individual tokens. Other work incorporates higher-level structure while retaining token prediction: Chen et al. (2025) use multi-token prediction during posttraining to encourage learning of coherent multitoken units, Tack et al. (2025) combine NTP with continuous concept representations derived from model activations, and Liu et al. (2026) jointly predict discrete latent concepts and tokens. These approaches differ in what constitutes a “concept,” but share the goal of providing supervision above the level of individual tokens.

Most closely related to our setting, Iyer et al. (2026) incorporate lexical-semantic concepts into post-training using synonyms and hypernyms through both data augmentation and concept-level objectives. Zhang et al. (2026) instead construct self-supervised sets of contextually equivalent tokens over nouns, verbs, and adjectives and directly optimize probability mass over those concept sets. Our question is complementary: we retain the standard NTP objective and use only data augmentation, allowing us to isolate how the part of speech carrying the concept signal affects perplexity, downstream transfer, training dynamics, and representation geometry.

![](images/afdbecceaf3aff125aaa7d6a9c379bdb5dd15e18235a1fb06a5bae4cca0962e8.jpg)  
Figure 2: Overview of the concept-augmentation and evaluation pipeline. Datapoints are drawn from arXiv, news, YouTube, or the combined corpus, and target words are selected by part of speech. For each POS condition, training data are either left unchanged (vanilla) or augmented with context-aware or dictionary-based synonym or hypernym substitutions. Each resulting corpus is used to further train Llama-3-8B with the standard next-token prediction objective, producing 96 post-trained models, which are subsequently evaluated through LoRA-based fine-tuning on seven downstream classification tasks.

## 3 Data Processing and Models

We follow the concept extraction, data augmentation, and post-training pipeline introduced by Iyer et al. (2026). Our downstream fine-tuning procedure differs and is described in Section 4.2. We therefore summarize the original pipeline and focus on the modifications introduced in this study.

## 3.1 Data & Concept Supervision Extraction

We use the post-training corpora of Iyer et al. (2026): arXiv abstracts, news text, and YouTube comments, representing scientific, journalistic, and informal language, respectively, as well as their combined condition. These domains let us test whether the effectiveness of POS-specific concept supervision depends on the domain in which it is introduced. We downsample conditions where necessary to balance post-training volume across augmentation variants.

We reprocess each corpus to extend the original noun-only augmentation to four target conditions: nouns, verbs, adjectives, and all POSs, combining all three categories.

For each target POS, we retain the two levels of conceptual resolution from Iyer et al. (2026): synonyms, which provide alternatives at approximately the same semantic resolution (e.g., coach → trainer), and hypernyms, which provide broader conceptual abstractions (e.g., coach → professional).

We also retain their two extraction procedures: context-aware alternatives generated by an LLM conditioned on the full sentence and target word, and dictionary-based alternatives extracted from WordNet without sentence context.

Figure 1 depicts all of these conditions.

## 3.2 Concept Models

All experiments initialize from Llama-3-8B (Grattafiori et al., 2024). We further post-train a separate model for each POS, post-training corpus, semantic relation, and concept-source condition described above, following the post-training setup of Iyer et al. (2026).

The original study explored two ways to insert concept-supervision during post-training: a modified concept-level training objective and conceptaware data augmentation, in which additional training sentences are created using conceptually related substitutions. Because the data-augmentation approach performs more consistently in the original study, we focus exclusively on this setting. Figure 2 summarizes the full training pipeline, from POS-specific augmentation through standard NTP post-training and downstream fine-tuning.

## 3.3 Baselines

We post-train each POS condition on arXiv, news, YouTube, and a combined corpus. Each augmented model is paired with a size-matched vanilla model constructed from the same source sentences. Instead of replacing the target with a synonym or hypernym, the vanilla dataset repeats the original sentence as needed to match the number of augmented training examples. We construct this baseline separately for each POS, relation, and corpus, so augmentation and vanilla models differ in conceptual substitution rather than total training volume.

![](images/74d5adfb8bcfc2e8c2908eaa2294be21c800262a32df4c28e92e65d2d1cba4df.jpg)  
Figure 3: The effect of concept augmentation on perplexity depends on both post-training corpus and POS. Perplexity improvement of concept-augmented models relative to their matched vanilla baselines, separated by post-training corpus. Positive values indicate lower perplexity after augmentation. Points show context-aware and dictionary-based augmentation for nouns (N), verbs (V), adjectives (Adj), and all three parts of speech (ALL); dotted horizontal lines show the mean improvement for each corpus. Augmentation improves perplexity most consistently when post-trained on news (+5.5%), while it decreases performance on arXiv (−8.2%). The relative effect of each POS also changes substantially across corpora.

## 4 Experimental Setup

We now describe the evaluation procedure for the post-trained models. We measure intrinsic language-model performance using held-out perplexity and downstream transfer using seven classification benchmarks, reporting all results relative to matched vanilla models where applicable.

## 4.1 Perplexity

For perplexity, we report the absolute improvement relative to the matched vanilla model:

$$
\mathrm { P P L } \mathrm { I m p r o v e m e n t } = \mathrm { P P L } _ { \mathrm { v a n i l l a } } - \mathrm { P P L } _ { \mathrm { c o n c e p t } } .\tag{1}
$$

Positive values indicate lower, and therefore better, perplexity than the matched vanilla model.

For downstream evaluation, we report the accuracy difference from the corresponding vanilla model:

$$
\Delta _ { \mathrm { A c c } } = \mathrm { A c c _ { c o n c e p t } - A c c _ { v a n i l l a } } .\tag{2}
$$

This lets us measure what conceptual augmentation contributes beyond additional exposure to the same underlying data.

## 4.2 Downstream Evaluation

We evaluate the post-trained models on the same seven downstream classification tasks as Iyer et al. (2026), but replace the original instruction-style fine-tuning procedure with LoRA-based sequence classification (Hu et al., 2021). This gives every benchmark a common classification formulation and enables validation-based early stopping, which we additionally use to analyze fine-tuning dynamics. We select the checkpoint with the lowest validation loss and report held-out test accuracy as our primary metric, along with weighted precision, recall, F1, and AUC-PR. Full LoRA and optimization hyperparameters are provided in Appendix C.

## 5 Results

We first evaluate how POS-specific concept augmentation affects model performance, beginning with a reproducibility check against the noun-only results of Iyer et al. (2026). We then examine heldout perplexity, the relationship between perplexity and downstream transfer, and performance across the seven downstream tasks.

## 5.1 Reproducibility Reality Check

Before comparing POS conditions, we verify that our pipeline recovers the noun-based behavior reported by Iyer et al. (2026). Our noun-only results reproduce the same broad perplexity and downstream trends, providing a sanity check for the expanded experiments. Small differences are expected because we reprocess the post-training data to extract multiple POS categories and use a revised LoRA-based downstream fine-tuning procedure. The full comparison is described in Appendix A.

![](images/98eb98910f0a04c31a53a565b2760674487a3cbd7883900d2593d35801b4048f.jpg)  
Figure 4: Change in language-model quality against change in downstream performance, each measured relative to the model’s own vanilla counterpart. Each point is one post-trained model; the x-axis is the reduction in perplexity averaged over the four evaluation corpora, and the y-axis is the gain in accuracy averaged over the seven downstream tasks. Points in the shaded upper-right quadrant improved on both measures. Marker shape and colour denote the substituted POS; filled markers are hypernym substitution and open markers denote synonym substitution. The correlation is $r = + 0 . 3 8 ( p = 0 . 0 3 3 )$ for context augmentation and $r = + 0 . 3 0 ( p = 0 . 0 9 6 )$ for dictionary augmentation.

## 5.2 Perplexity

We first ask whether the effect of concept augmentation on perplexity changes across post-training domains and POSs. Figure 3 reports each augmented model’s improvement relative to its matched vanilla model, where positive values indicate lower perplexity.

The clearest effect is driven by the post-training corpus. Concept augmentation improves perplexity for models post-trained on news, with an average improvement of 5.5%, while it hurts models post-trained on arXiv by 8.2%. Models trained on YouTube and the combined corpus show smaller average decreases of 2.6% and 2.5%, respectively. Thus, the same augmentation procedure can improve or worsen perplexity depending on the domain in which it is introduced.

Within each corpus, we also observe a strong interaction with POS. On arXiv, augmenting all POSs is the only condition that improves perplexity on average; among the individual POS conditions, verbs degrade substantially less than nouns or adjectives. News shows a different pattern: augmentation is generally beneficial, with adjectives producing the largest gains. For hypernyms, adjective and verb augmentation reduce perplexity by 14.4% and 13.8%, respectively, relative to their vanilla baselines. On YouTube, verbs are instead the clearest individual-POS beneficiary: synonym augmentation improves perplexity by 12.6%, while adjective performance is much more sensitive to the concept source. The combined corpus again produces a different ordering, with adjective hypernyms improving by 9.8% but adjective synonyms substantially worsening perplexity.

Overall, there is no single POS whose augmentation consistently lowers perplexity. Rather, the effect of POS is conditioned on both the posttraining domain and the semantic relation being augmented. This domain dependence motivates looking beyond perplexity alone when evaluating whether the resulting models learn more useful concept representations.

## 5.3 Perplexity and Downstream

Figure 4 compares intrinsic and extrinsic effects by plotting each model’s perplexity improvement against its downstream accuracy gain, both measured relative to the matched vanilla baseline. The overall relationship is positive but only moderate. For context augmentation, the correlation between average perplexity improvement and average downstream accuracy gain is $r = 0 . 3 8 ( p = 0 . 0 3 3 )$ ; for dictionary augmentation, it is weaker at r = 0.30 $( p = 0 . 0 9 6 )$ . Thus, lower perplexity tends to accompany better downstream performance, but the relationship remains noisy.

(a) Accuracy change vs vanilla, by part of speech
<table><tr><td>HYyprrnym</td><td>Arxiv</td><td>-3.7</td><td>+4.5</td><td>-3.4</td><td>-3.7</td></tr><tr><td></td><td>News</td><td>-1.4</td><td>+2.9</td><td>-0.4</td><td>+0.2</td></tr><tr><td>Combined</td><td>YouTube</td><td>+2.1</td><td>-1.5</td><td>+1.3</td><td>+2.0</td></tr><tr><td colspan="2">Synonym</td><td>-0.1</td><td>+0.5</td><td>-0.7</td><td>-0.6</td></tr><tr><td></td><td>Arxiv</td><td>-2.7</td><td>+2.2</td><td>-0.3</td><td>+1.6</td></tr><tr><td></td><td>News</td><td>-1.1</td><td>-2.4</td><td>+3.4</td><td>+0.6</td></tr><tr><td></td><td>YouTube</td><td>-1.4</td><td>+2.8</td><td>-2.8</td><td>-0.4</td></tr><tr><td></td><td>Combined</td><td>+0.7</td><td>+0.9</td><td>-0.5</td><td>-0.9</td></tr><tr><td colspan="4">Nouns Verbs</td><td>Adj</td><td>All</td></tr></table>

![](images/9b6085b33058c7d24675b190cf45b422341ac71e2177cbfbc8051d3fa3ac5afe.jpg)  
Figure 5: Downstream effects depend on both the augmented POS and evaluation task. (a) Mean accuracy change relative to the matched vanilla baseline for each semantic-relation×corpus×POS condition, averaged over the seven downstream tasks and the context-aware and dictionary-based variants. Positive values indicate improvement over vanilla; outlined cells mark the best POS within each row. Verbs achieve the strongest mean performance in six of the eight settings. (b) Range of accuracy changes across augmentation conditions for each downstream task. GLUE, SNLI, and Logical Fallacy are substantially more sensitive to the augmentation choice than Spam, Hate Speech, and Emotion.

Models appear in all four quadrants of Figure 4, indicating that better language-modeling performance does not reliably translate into better downstream accuracy, and vice versa. Some settings improve downstream performance despite little or no perplexity gain, while others reduce perplexity without producing corresponding downstream improvements. Together with the results in Section 5.2, where the effect of augmentation on perplexity varies substantially across post-training corpora and POS conditions, this suggests that perplexity is only a partial proxy for the usefulness of concept-based augmentation.

We therefore treat downstream task performance as the primary criterion in the remainder of our analysis, using perplexity as a complementary intrinsic signal.

## 5.4 Downstream Performance

Because perplexity does not yield a single consistent ranking of augmentation strategies, we next focus on downstream task performance. Appendix D includes tables that summarize average accuracy changes relative to vanilla baselines, isolating the effects of part of speech and augmentation variant,

respectively.

A first result is that verbs are the most consistently effective POS. Averaging over contextaware and dictionary-based variants, verb augmentation gives the best mean result in six of the eight semantic-relation×corpus settings: hypernym arXiv (+4.5), hypernym news (+2.9), hypernym combined (+0.5), synonym arXiv (+2.2), synonym YouTube (+2.8), and synonym combined (+0.9). The pattern is not universal: hypernym YouTube is strongest for nouns, while synonym news is strongest for adjectives (+3.4). Nevertheless, verbs are the most consistently competitive single-POS condition overall. Augmenting all POSs simultaneously does not consistently outperform the strongest targeted POS condition.

Downstream gains are also highly taskdependent. Figure 5b shows substantial variation across tasks and augmentation conditions. The largest swings occur on GLUE, including gains for hypernym verbs on arXiv (+30.9), synonym verbs on arXiv (+23.2), synonym adjectives on news (+28.8), and hypernym all-POS on YouTube (+15.0). Logical Fallacy is also sensitive to augmentation choice, whereas Spam, Hate Speech, and Emotion are comparatively stable.

The augmentation source provides a second axis of variation. Averaging across POS conditions, dictionary-based hypernym augmentation outperforms context-aware augmentation on news (+1.0 vs. −0.3), YouTube (+1.9 vs. 0.0), and the combined corpus (+0.2 vs. −0.7), with arXiv as the exception. For synonym augmentation, the preferred source depends on the corpus: dictionary-based augmentation is stronger on arXiv and the combined corpus, while context-aware augmentation is stronger on news and YouTube.

Overall, there is no globally best augmentation strategy. Downstream performance instead reflects an interaction among the post-training corpus, semantic relation, concept source, and POS being targeted.

## 6 Analyses

To better understand the behavioral differences observed above, we analyze how concept augmentation affects both downstream adaptation and the model’s internal representations. We first examine fine-tuning dynamics to see how quickly models adapt to downstream tasks and then use CKA and linear probing to characterize how the different augmentation conditions alter representation geometry.

## 6.1 Training Dynamics

The downstream results show that POS and posttraining corpus affect final accuracy differently. We next ask whether they also affect how models adapt during downstream fine-tuning. We examine two aspects of the training trajectory: the step at which fine-tuning stops and the early rate of improvement. To make these quantities comparable across tasks with different difficulty and accuracy ranges, both are standardized within each downstream task.

![](images/5ae0bd4fade2a2cccab65392da45b97b030ad58e712b6c7da550ef6d11af4bc1.jpg)  
Figure 6: POS affects when downstream fine-tuning stops. Mean standardized stopping point across downstream tasks, grouped by the POS augmented during post-training. Adjective-augmented models stop significantly earlier than all-POS models $( p = 0 . 0 0 1 )$ , indicating that the breadth of the concept signal affects downstream convergence.

Figure 6 shows that the POS used for concept augmentation affects when fine-tuning converges. Models augmented over adjectives stop significantly earlier than models augmented over all POSs $( p = 0 . 0 0 1 )$ ). This suggests that exposing the model to concept variation across several grammatical categories may require more downstream adaptation than restricting the concept signal to adjectives alone.

![](images/ce952a80bd87a941faa906b126c28cfa3f3e44841467fd4adfb4aaae8b237674.jpg)  
Figure 7: Post-training corpus affects early finetuning speed. Standardized accuracy gain between fine-tuning steps 250 and 500, grouped by post-training corpus. Models post-trained on the combined corpus improve significantly faster over the first interval than models post-trained on YouTube $( p = 0 . 0 0 4 )$ .

The post-training corpus instead affects the early rate of fine-tuning. Figure 7 measures the accuracy gain between steps 250 and 500. Models post-trained on the combined corpus improve significantly faster over this interval than models post-trained on YouTube $( p = 0 . 0 0 4 )$ . Thus, the factors controlling when training stops and how quickly performance initially improves are not identical: POS is associated with the convergence point, while corpus is associated with early learning speed.

To probe these differences, we compare the semantic coherence of concept sets using pairwise cosine similarity within and across sets by POS and corpus. This lets us test whether differences in concept-set structure align with the observed convergence and learning-rate effects.

## 6.2 Representational Analysis

CKA shows that representational change depends strongly on POS $( p < 0 . 0 0 1 )$ . At the final layer, all-POS models diverge most from the base model (CKA = 0.743), while verb models remain most similar (0.810), with nouns (0.796) and adjectives (0.790) in between. Post-training corpus also matters $( p = 0 . 0 0 4 )$ : arXiv models show the largest divergence (0.751), while the other corpora remain near 0.79-0.80. Greater divergence is associated with lower further-training loss $( \rho = - 0 . 3 8 , p =$ 0.008), but higher held-out perplexity $( \rho = 0 . 3 5$ $p = 0 . 0 1 5 )$

Linear probing shows a related but weaker pattern. POS information generally becomes more linearly decodable in later layers, but much of this increase is also present in matched vanilla models. All-POS augmentation shows the clearest conceptspecific probe effect: dictionary models improve probe accuracy by approximately 0.17 percentage points over vanilla across layers, reaching approximately +0.45 points at the final probed layer. Thus, both CKA and probing identify all-POS augmentation as producing the largest representational change, even though all-POS is not the strongest downstream condition.

## 7 Discussion

Our results show that the usefulness of concept augmentation depends on where the concept signal is introduced. Verbs provide the most consistent downstream gains, while the strongest POS still varies across domains and tasks. In contrast, augmenting all three POS categories together does not reliably outperform targeted augmentation. This suggests that concept supervision is not simply additive. Different POS categories encode different kinds of semantic structure, and exposing the model to more conceptual variation does not necessarily produce more useful representations. One possible explanation for the strength of verbs is that they encode actions and relations that organize the event structure of a sentence, so varying their word choice may encourage invariance to surface form while preserving relational meaning.

The perplexity and CKA results further suggest a distinction between fitting the post-training distribution and preserving transferable structure. Models that move further from the pretrained representation achieve lower further-training loss, but also have higher held-out perplexity. At the same time, the conditions that produce the largest representational change are not the conditions that transfer best: all-POS models diverge most under CKA, whereas verb models remain comparatively close to the base model while providing stronger downstream performance. This pattern is consistent with post-training specialization, where stronger fitting to the observed next-token distribution need not preserve the structure most useful outside that distribution (Bachmann and Nagarajan, 2025).

Linear probing provides a complementary result. POS information becomes more linearly decodable after further training, but much of this effect also appears in matched vanilla models. All-POS augmentation produces the clearest concept-specific probe increase, yet does not yield the strongest downstream performance. Conversely, verb augmentation transfers well without producing unusually large POS-probe gains. The probe therefore appears to operate at too coarse a level of resolution to explain the behavioral differences. Concept augmentation may instead change semantic structure within a POS, such as event structure among verbs, entity categories among nouns, or attribute structure among adjectives. Together, the CKA and probing results suggest that the usefulness of concept augmentation depends less on how much the representation changes than on what information changes. This motivates future concept-aware objectives that condition supervision on linguistic category rather than treating all content words uniformly.

## 8 Conclusion

We study whether the part of speech carrying concept supervision changes the effectiveness of concept-aware data augmentation. Extending prior noun-focused experiments to verbs, adjectives, and all three POS categories, we find that augmentation effects depend strongly on both POS and posttraining domain. Verb augmentation provides the most consistent downstream gains, while augmenting all POSs together does not reliably outperform targeted augmentation.

Our analyses further show that larger representational changes are not necessarily more useful. All-POS models change most under CKA and linear probing, whereas verb models transfer more consistently while remaining closer to the pretrained representation. These findings motivate POS-aware concept supervision, where conceptual variation is designed around the type of semantic structure a linguistic category contributes rather than applied uniformly across tokens.

## Limitations

Our experiments are limited to Llama-3-8B and English-language corpora, so the interaction between POS and concept augmentation may differ across model scales, architectures, and languages. We also study data augmentation only and therefore do not establish whether the same POS-specific patterns hold under direct concept-level objectives.

Additionally, concept extraction quality varies across parts of speech and semantic relations. Hypernymy is particularly less well-defined for adjectives than for nouns and verbs, and context-aware generation and dictionary extraction introduce different forms of noise. These differences may partly explain the domain- and POS-specific effects observed here.

## Ethical Considerations

Concept-aware data augmentation can introduce distortions where synonyms or hypernyms are not interchangeable in context. In sensitive contexts, such substitutions could also amplify stereotypes or spurious associations already present in the training data. Care should therefore be taken when constructing and inspecting concept sets, especially for demographic or otherwise sensitive content.

Our results also show that the effects of concept augmentation vary substantially across POS, domains, and downstream tasks. Improvements in aggregate perplexity or accuracy therefore should not be interpreted as evidence of improved fairness, safety, or reliability. We do not evaluate toxicity, demographic bias, hallucination, or other deployment-related harms in this work. Practitioners using concept augmentation should validate both the generated training examples and resulting model behavior before deployment.

Disclosure: The authors used LLMs to assist with revising portions of the manuscript and LaTeX formatting across the paper including tables. All results and claims were reviewed and verified by the authors.

## References

Gregor Bachmann and Vaishnavh Nagarajan. 2025. The pitfalls of next-token prediction. Preprint, arXiv:2403.06963.

Michael K. Chen, Xikun Zhang, Jiaxing Huang, and Dacheng Tao. 2025. Improving large language models with concept-aware fine-tuning. Preprint, arXiv:2506.07833.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Laya Iyer, Pranav Somani, Alice Guo, Dan Jurafsky, and Chen Shani. 2026. Beyond tokens: Conceptlevel training objectives for LLMs. In Proceedings of the 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 457–474, Rabat, Morocco. Association for Computational Linguistics.

Yuliang Liu, Yunchong Song, Yixuan Wang, Kewen Ge, Alex Lamb, Qipeng Guo, Kai Chen, Bowen Zhou, and Zhouhan Lin. 2026. Next concept prediction in discrete latent space leads to stronger language models. Preprint, arXiv:2602.08984.

Chen Shani, Jilles Vreeken, and Dafna Shahaf. 2023. Towards concept-aware large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13158–13170, Singapore. Association for Computational Linguistics.

Jihoon Tack, Jack Lanchantin, Jane Yu, Andrew Cohen, Ilia Kulikov, Janice Lan, Shibo Hao, Yuandong Tian, Jason Weston, and Xian Li. 2025. Llm pretraining with continuous concepts. Preprint, arXiv:2502.08524.

LCM team, Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R. Costa-jussà, David Dale, Hady Elsahar, Kevin Heffernan, João Maria Janeiro, Tuan Tran, Christophe Ropers, Eduardo Sánchez, Robin San Roman, Alexandre Mourachko, and 2 others. 2024. Large concept models: Language modeling in a sentence representation space. Preprint, arXiv:2412.08821.

Christine Zhang, Dan Jurafsky, and Chen Shani. 2026. Learning concepts, not tokens: Self-supervised semantic alignment for language models. Preprint, arXiv:2603.29123.

![](images/844246b886303f88b29dd5954b2b6e9ba17eb610cef81e82800749c8479b1af0.jpg)

![](images/ab5e41724b636c761cad782de2118e638afa181e37b9fe2972d71de3b75d9026.jpg)  
Figure 8: Reproducibility check against the noun-only experiments of Iyer et al. (2026). Comparison of the original results with our noun-based conditions. Left: perplexity, colored by post-training corpus. Right: downstream accuracy, colored by task. The diagonal indicates identical performance. Our pipeline recovers the same broad trends, with small differences due to data reprocessing and the revised downstream fine-tuning procedure.

## A Reality Check

Our noun-only experiments recover the broad behavior reported by Iyer et al. (2026), providing a sanity check before extending the augmentation procedure to additional POS categories as shown in Figure 8. Our perplexities are generally somewhat lower than those in the original study. We reprocess each corpus to identify noun, verb, and adjective targets jointly, which changes the set and number of eligible noun targets relative to the original nounonly pipeline. Downstream differences may additionally reflect our use of LoRA-based sequence classification rather than the original downstream fine-tuning procedure.

## B Additional Perplexity Results

The main paper reports perplexity relative to matched vanilla models in order to isolate the contribution of concept augmentation. Figure 9 provides the corresponding absolute perplexities and shows how each post-training condition generalizes across evaluation domains.

## C Downstream Fine-Tuning Details

We evaluate the post-trained models on the same seven downstream classification tasks as Iyer et al. (2026). We formulate each benchmark as sequence classification and adapt the post-trained model using LoRA (Hu et al., 2021), training a randomly initialized classification head together with lowrank adapters while keeping the original backbone weights frozen.

This parameter-efficient setup is useful for our comparison because the experiment contains 96 post-trained models evaluated across seven tasks. Restricting downstream adaptation to LoRA parameters and the classification head substantially reduces the number of trainable parameters and provides a consistent fine-tuning procedure across all model-task combinations. It also limits modification of the post-trained backbone, making downstream comparisons more reflective of differences induced during post-training.

We use LoRA with rank $r \ = \ 1 6 , \ \alpha \ = \ 3 2 .$ dropout 0.05, and adapters on the q\_proj and v\_proj projections. Unless pre-existing splits are available, each dataset is divided into 70% training, 15% validation, and 15% test data using split seed 42. Inputs are truncated and padded to 256 tokens. We train with learning rate $2 \times 1 0 ^ { - 5 }$ , per-device batch size 2, gradient accumulation over four steps, weight decay 0.01, and bfloat16 precision for at most 2,000 steps.

We evaluate on the validation set every 20 steps and apply early stopping with patience 3 and minimum improvement threshold 0.001, selecting the checkpoint with the lowest validation loss. This allows different post-trained models to converge at different rates rather than forcing a fixed number of downstream updates. We report held-out test accuracy as our primary metric together with weighted precision, recall, F1, and AUC-PR.

![](images/c7ecd47bcb5c1f0cd086265cf13b4b632746b19352de2c9c137ec19148f6ee8a.jpg)

Figure 9: Absolute perplexity across post-training and evaluation corpora. Each panel corresponds to one post-training corpus, and the x-axis denotes the evaluation corpus. Co lor denotes the augmented POS. Solid lines show concept-augmented models and dotted lines show their corresponding vanilla conditions. Values are averaged across synonym and hypernym settings.
<table><tr><td>Relation</td><td>Post-training corpus</td><td>Nouns</td><td>Verbs</td><td>Adjectives</td><td>All POS</td></tr><tr><td rowspan="4">Hypernym</td><td>arXiv</td><td>-3.7</td><td>+4.5</td><td>-3.4</td><td>-3.7</td></tr><tr><td>News</td><td>-1.4</td><td>+2.9</td><td>-0.4</td><td>+0.2</td></tr><tr><td>YouTube</td><td>+2.1</td><td>-1.5</td><td>+1.3</td><td>+2.0</td></tr><tr><td>Combined</td><td>-0.1</td><td>+0.5</td><td>-0.7</td><td>-0.6</td></tr><tr><td rowspan="4">Synonym</td><td>arXiv</td><td>-2.7</td><td>+2.2</td><td>-0.3</td><td>+1.6</td></tr><tr><td>News</td><td>-1.1</td><td>-2.4</td><td>+3.4</td><td>+0.6</td></tr><tr><td>YouTube</td><td>-1.4</td><td>+2.8</td><td>-2.8</td><td>-0.4</td></tr><tr><td>Combined</td><td>+0.7</td><td>+0.9</td><td>-0.5</td><td>-0.9</td></tr></table>

Table 1: Downstream accuracy change by augmented POS. Mean accuracy change in percentage points relative to matched vanilla models, separated by semantic relation, post-training corpus, and augmented POS. Context-aware and dictionary-based variants are averaged. Bold denotes the strongest POS within each relation-corpus condition. Verb augmentation is strongest in six of the eight conditions.

## D Downstream Results

<table><tr><td>Relation</td><td>Corpus</td><td>Context</td><td>Dictionary</td></tr><tr><td rowspan="4">Hypernym</td><td>arXiv</td><td>-1.1</td><td>-2.1</td></tr><tr><td>News</td><td>-0.3</td><td>+1.0</td></tr><tr><td>YouTube</td><td>0.0</td><td>+1.9</td></tr><tr><td>Combined</td><td>-0.7</td><td>+0.2</td></tr><tr><td rowspan="4">Synonym</td><td>arXiv</td><td>-1.5</td><td>+1.9</td></tr><tr><td>News</td><td>+0.8</td><td>-0.5</td></tr><tr><td>YouTube</td><td>-0.3</td><td>-0.7</td></tr><tr><td>Combined</td><td>-0.1</td><td>+0.2</td></tr></table>

Table 2: Downstream accuracy change by augmentation source. Mean accuracy change in percentage points relative to matched vanilla models, separated by semantic relation, post-training corpus, and concept source. Results are averaged across POS conditions and downstream tasks. Bold denotes the stronger source within each row.

Table 1 summarizes the POS comparison used in Section 4.2. Values are accuracy changes relative to the matched vanilla model, averaged across the seven downstream tasks and the context-aware and dictionary-based augmentation variants.

Table 2 instead averages across POS conditions to isolate the effect of the concept source.

## E Training-Dynamics Details

We analyze two properties of downstream finetuning trajectories: the stopping point selected by validation-based early stopping and the early accuracy improvement between steps 250 and 500. Because downstream tasks differ substantially in difficulty and absolute accuracy, both quantities are standardized within task before comparisons are made.

For stopping point, adjective-augmented models terminate earlier than all-POS models (p = 0.001); verb and adjective conditions also differ $( p = 0 . 0 0 6 )$ . We do not observe corresponding pairwise differences in stopping point across posttraining corpora.

<table><tr><td>Post-training corpus</td><td>Nouns</td><td>Verbs</td><td>Adjectives</td><td>All POS</td><td>Mean</td></tr><tr><td>arXiv</td><td>0.750</td><td>0.788</td><td>0.768</td><td>0.696</td><td>0.751</td></tr><tr><td>News</td><td>0.811</td><td>0.825</td><td>0.794</td><td>0.770</td><td>0.800</td></tr><tr><td>YouTube</td><td>0.809</td><td>0.820</td><td>0.791</td><td>0.746</td><td>0.792</td></tr><tr><td>Combined</td><td>0.814</td><td>0.807</td><td>0.807</td><td>0.761</td><td>0.797</td></tr><tr><td>Mean</td><td>0.796</td><td>0.810</td><td>0.790</td><td>0.743</td><td>0.785</td></tr></table>

Table 3: Final-layer CKA with the base model by POS and post-training corpus. Values are averaged across context-aware, dictionary-based, and matched vanilla hypernym conditions. Lower values indicate greater representational divergence. All-POS models show the greatest divergence, particularly after arXiv post-training, while verb models remain closest to the base model.

For early fine-tuning rate, the clearest corpuslevel difference is between the combined and YouTube post-training conditions: combinedcorpus models improve more rapidly between steps 250 and 500 $( p = 0 . 0 0 4 )$ . We do not observe a comparably strong POS-level difference in early slope. These results motivate separating the factors associated with when fine-tuning converges from those associated with how quickly accuracy initially improves.

## F Representational Analyses

## F.1 Centered Kernel Alignment

We use Centered Kernel Alignment (CKA) to compare the hidden representations of each post-trained model with those of the untouched Llama-3-8B base model. The aggregate CKA analysis reported here covers the hypernym conditions. Higher CKA indicates more similar representation geometry, while lower CKA indicates greater divergence from the base model.

Table 3 shows that final-layer similarity depends strongly on the POS targeted during augmentation. All-POS models diverge most from the base model, while verb models remain most similar. The posttraining corpus also matters, with arXiv producing the greatest average divergence.

The same pattern is visible when separating augmentation source (Table 5). Dictionary-based conditions have the lowest mean CKA overall, while the direction of this effect differs by POS.

Across models, greater representational divergence is associated with lower further-training loss $( \rho = - 0 . 3 8 , p = 0 . 0 0 8$ when CKA similarity is correlated with loss in the reported direction) and higher held-out perplexity $( \rho = 0 . 3 5 , p = 0 . 0 1 5 )$ Together with the downstream results, this suggests

that larger representational changes are not necessarily more transferable.
<table><tr><td>Variant</td><td>N</td><td>V</td><td>Adj</td><td>All</td><td>Mean</td></tr><tr><td>Vanilla</td><td>0.821</td><td>0.805</td><td>0.799</td><td>0.760</td><td>0.796</td></tr><tr><td>Context</td><td>0.805</td><td>0.807</td><td>0.789</td><td>0.743</td><td>0.786</td></tr><tr><td>Dict.</td><td>0.763</td><td>0.817</td><td>0.783</td><td>0.728</td><td>0.773</td></tr></table>

Table 5: Final-layer CKA with the base model by augmentation source. Values are averaged across post-training corpora for the hypernym conditions. Dictionary-based models have lower mean CKA than their matched vanilla models, although the effect is not uniform across POS categories.

## F.2 Linear Probing

We train linear probes at layers 0, 4, 8, 12, 16, 20, 24, 28, and 32 to predict whether token representations correspond to nouns, verbs, or adjectives. Probe accuracy generally increases in later layers after further training. However, matched vanilla models show the same broad pattern, indicating that much of the increase in POS decodability is attributable to further training rather than concept augmentation specifically.

<table><tr><td>Layer</td><td>Context</td><td>Dictionary</td><td>Vanilla</td></tr><tr><td>0</td><td>-0.06</td><td>-0.08</td><td>-0.08</td></tr><tr><td>4</td><td>-0.10</td><td>-0.06</td><td>-0.10</td></tr><tr><td>8</td><td>+0.15</td><td>+0.27</td><td>+0.25</td></tr><tr><td>12</td><td>+1.12</td><td>+1.23</td><td>+1.23</td></tr><tr><td>16</td><td>+0.31</td><td>+0.32</td><td>+0.33</td></tr><tr><td>20</td><td>+0.71</td><td>+0.76</td><td>+0.76</td></tr><tr><td>24</td><td>+0.93</td><td>+1.03</td><td>+1.01</td></tr><tr><td>28</td><td>+1.22</td><td>+1.44</td><td>+1.27</td></tr><tr><td>32</td><td>+1.46</td><td>+1.76</td><td>+1.52</td></tr></table>

Table 6: Mean POS-probe accuracy change relative to the untouched base model. Values are percentagepoint changes averaged over POS targets, corpora, and semantic relations. All three variants become more POSdecodable in later layers, including the matched vanilla models.

<table><tr><td rowspan="2">Target POS</td><td colspan="2">Mean across probed layers</td><td colspan="2">Final probed layer</td></tr><tr><td>Context</td><td>Dictionary</td><td>Context</td><td>Dictionary</td></tr><tr><td>Nouns</td><td>-0.04</td><td>+0.08</td><td>-0.12</td><td>+0.07</td></tr><tr><td>Verbs</td><td>+0.05</td><td>+0.02</td><td>+0.06</td><td>+0.10</td></tr><tr><td>Adjectives</td><td>-0.24</td><td>-0.05</td><td>-0.33</td><td>+0.36</td></tr><tr><td>Alí POS</td><td>+0.04</td><td>+0.17</td><td>+0.16</td><td>+0.45</td></tr></table>

Table 4: Concept-specific change in POS-probe accuracy relative to matched vanilla models. Values are percentage-point differences between the concept-augmented and matched vanilla probe accuracies. All-POS augmentation shows the clearest positive concept-specific effect, particularly for dictionary-based augmentation. These entries are differences of point estimates; bootstrap confidence intervals in the probe analysis are defined relative to the untouched base model rather than for the concept-minus-vanilla contrast.

Table 6 summarizes the mean change in probe accuracy relative to the untouched base model. The largest improvements occur in the later layers for all three post-training variants.

To isolate the additional effect of concept augmentation, we subtract the probe accuracy of the matched vanilla model from the context-aware and dictionary-based conditions. Table 4 reports these differences.

The probe results therefore support two conclusions. First, further post-training generally increases the linear accessibility of POS information, but this effect is not unique to concept augmentation. Second, all-POS augmentation produces the clearest additional probe effect relative to vanilla. This agrees with CKA in identifying all-POS models as changing most strongly, even though these models do not achieve the strongest downstream performance. Verb augmentation, by contrast, yields the most consistent downstream gains without a correspondingly large increase in coarse POS decodability.

## G Computational Resources

All experiments initialize from Llama-3-8B, which contains approximately 8 billion parameters. Posttraining updates the full model and uses five GPUs per run. Based on representative runs, post-training requires approximately 13 minutes per model, corresponding to approximately 103 GPU-hours across the 96 post-trained models.

Downstream evaluation uses LoRA-based sequence classification on a single NVIDIA A100- SXM4-80GB GPU. Each job evaluates one posttrained model across the seven downstream tasks and requires approximately 6.2 GPU-hours. Across the 96 models, this corresponds to approximately 595 GPU-hours.

In total, post-training and downstream finetuning require approximately 700 GPU-hours. Computational resources and GPUs were provided by the authors’ research institute.