# Which Forms of Caregiver Feedback Support Grammar Learning? A Reinforcement-Learning Study of Child-Like Language Models

Jing Liu\* Marianne Schweitzer\* Abdellah Fourtassi

ENS, Université PSL, EHESS, CNRS, Paris, France

jing.liu@psl.eu

Aix Marseille Université, CNRS, LIS, Marseille, France marianne.schweitzer@lis-lab.fr abdellah.fourtassi@lis-lab.fr

## Abstract

Social interaction is central to children’s language learning, but the effects of different forms of caregiver feedback are difficult to isolate in naturalistic data. We use child-like language models as controlled learners to test which forms of feedback support grammatical development. Small GPT-2-style models are pretrained on child-directed language from CHILDES, then fine-tuned with reinforcement learning using reward models trained to capture four feedback types: communicative feedback, structural alignment, semantic contingency, and affective feedback. Reward finetuning yields limited gains on minimal-pair evaluations, but clearer effects in free generation. Structural alignment produces the strongest improvements in grammaticality, providing a novel, plausible mechanistic account of how this feedback can support grammar learning. Communicative feedback yields more moderate gains. In contrast, semantic contingency and affective feedback do not improve grammaticality, although further analyses suggest that they may support other aspects of language learning beyond grammar. These results suggest that different forms of caregiver feedback make complementary contributions to language learning.

Keywords: child-directed language; language acquisition; reinforcement learning

## 1 Introduction

Children do not learn language from exposure to input alone, but also through social interaction in which caregivers respond to their produced utterances (Fusaroli et al., 2023; Masek et al., 2021; Clark, 2020). A long-standing question in language acquisition concerns the role of such feedback in shaping grammatical development, especially if caregiver responses can help the learner know when their productions were well- or ill-formed (see review in Sokolov and Snow, 1994).

Although caregivers rarely provide explicit correction of grammatical errors (Brown and Hanlon, 1970), researchers have proposed that they nevertheless provide global, noisy feedback from which children can infer whether their utterances were likely grammatical or ungrammatical (e.g., Hirsh-Pasek et al., 1984; Demetras et al., 1986; Sokolov and Snow, 1994). In this view, the child can be understood as learning in a reinforcement-like fashion, using the valence of the feedback as positive or negative rewards that guide which productions to maintain or adjust in future interactions (e.g., see Whitehurst and Valdez-Menchaca, 1988; Nikolaus and Fourtassi, 2023; Schoneberger, 2010).

The developmental literature has documented several forms of caregiver response that can be interpreted, within a reinforcement-learning framework, as valenced feedback. The most direct examples are communicative feedback cues, namely a) acknowledgments (e.g., "okay", "yeah"), which signal communicative success and thus increase the likelihood that the child’s sentence was wellformed, and b) clarification requests (e.g., "huh?", "what?"), which signal communicative failure or misunderstanding and thus increase the likelihood that the sentence was ill-formed (Clark, 2020; Demetras et al., 1986; Newport et al., 1977; Saxton et al., 2005b; Nikolaus et al., 2023, 2022).

A second relevant class is semantic contingency, or the extent to which caregiver responses remain responsive to the child’s topic and intended meaning (Hoff-Ginsberg, 1987; Che et al., 2018; Hirsh-Pasek et al., 2015; Agrawal et al., 2024, 2026). The degree of this contingency captures the feedback valence: low contingency in the caregiver response signals semantic misalignment and thus suggests that the child’s utterance may be ill-formed.

There are other forms of caregiver feedback that also provide such valenced signals, even if they have not traditionally been associated with this learning framework. Perhaps most directly relevant to grammar learning is structural alignment (Fusaroli et al., 2023; Dale and Spivey, 2006; Fernandez and Grimm, 2014; Misiek et al., 2020), whereby caregivers respond by reusing the child’s syntactic frame while potentially introducing different lexical content (e.g., child: “I eat” (Pronoun-Verb) → caregiver: “Mommy eats too”). Here, the feedback valence is reflected in whether such alignment occurs, signaling support for the child’s use of that syntactic frame.

A final type of feedback is affective feedback, such as whether caregivers respond with positive affect (e.g., "that’s amazing!") versus a more emotionally neutral response (Rivero et al., 2023).

These cues are far more available to children than explicit grammatical corrections, making their role in language learning much more plausible (Demetras et al., 1986; Hirsh-Pasek et al., 1984). Yet, they provide only indirect and noisy learning signals. First, they are not always directed at grammar specifically. Second, even in cases when grammar is the main target, the signal is global (i.e., utterance-level) and does not identify the specific error within the utterance. This makes credit assignment a central challenge (Marcus, 1993; Pinker, 1989; Saxton, 1997), and also makes interactive learning from caregiver feedback an interesting empirical case.

Nevertheless, little is known about how these different feedback types contribute to children’s grammatical learning. While corpus studies allow a quantitative account of caregivers’ feedback behavior, the causal effect of each type of feedback or reward on learning remains difficult to establish. A common methodological issue facing previous attempts (e.g., see discussion in Saxton et al., 2005a) is that different types of feedback are provided to the same child, making it difficult—and ethically problematic—to isolate their effect over the developmental timescales on which language is learned.

Recent computational work has begun to revisit these questions by using LMs as controlled learning environments for testing hypotheses about language learning, thereby combining the targeted precision of experimental methods with the ecological validity of corpus studies (e.g., Misra and Mahowald, 2024). Particularly relevant to our question are studies that used Reinforcement Learning to investigate LMs’ interactive learning strategies (Ma et al., 2024; Mayer Martins et al., 2025; Zhao

et al., 2023; Padovani et al., 2025; Xu et al., 2025;   
Nikolaus and Fourtassi, 2021).

Recent work has applied this approach to natural child–caregiver interactions, using Reward Models trained from caregivers’ actual feedback data (Nikolaus and Fourtassi, 2026). Language models are then fine-tuned using techniques inspired by Reinforcement Learning from Human Feedback (RLHF, Ouyang et al., 2022). This work showed that caregiver-like clarification requests on model generation can signal negative evidence. Learning from this signal improved grammaticality.

## The current study

Existing RLHF-based studies of naturally occurring caregiver feedback have focused primarily on clarification requests, leaving open whether other forms of social feedback have similar, weaker, or even negative effects on grammatical learning. The present study addresses this gap by comparing several theoretically motivated feedback types within the same controlled learning framework.

## 2 Methods

Following the broad methodology outlined in Nikolaus and Fourtassi (2026), we use a two-stage learning setup designed to separate learning from linguistic exposure from learning driven by caregiverfeedback rewards. In the first step, a small causal language model is pretrained on child-directed language (excluding children’s production), providing an input-only baseline that quantifies learning gains from linguistic exposure. In a second step, the model is fine-tuned via RL, using a reward model trained on the target caregiver’s feedback valence. This design allows us to isolate factors that typically co-vary in children’s experience; providing an opportunity to test their specific effects on learning. Pretraining baselines We use a customized version of GPT-2 (2 layers, hidden size 512, 8 attention heads), pretrained on CHILDES (MacWhinney, 2000), using three pretraining scales (0.1M, 1M, and 10M words, see Appendix 4).

Empirical Reward Models For each feedback type, we trained a reward model to capture naturally occurring patterns of caregiver feedback in child– caregiver interactions (using CHILDES dataset). Each reward model was trained to map children’s utterances to a scalar, representing the valence of the target feedback (see Appendix 4). The valence values themselves were annotated prior to rewardmodel training using a range of automatic annotation tools (see below, and Appendix 3).

Topline Reward Model We use an idealized reward that directly targets grammatical errors. This topline is not intended to produce empirically realistic results, but rather to serve as an upper bound on what grammatical learning is possible in principle (see details in Appendix 4).

RL fine-tuning For each feedback/reward, the pretrained models were fine-tuned using the corresponding reward models, using Proximal Policy Optimization (PPO; Schulman et al., 2017). We included several safeguards to mitigate potential language drift (see Appendix 5).

Training seeds and variability To account for stochastic variability, we used, for each training scale, a fully crossed 3 × 3 × 3 design over pretraining seeds, fine-tuning seeds, and generation seeds.

Annotation of the reward signals We considered the four classes of caregiver feedback mentioned in the introduction: communicative feedback, semantic contingency, structural alignment and affective feedback. For each type of feedback, we annotated the CHILDES child-caregiver conversations for the corresponding valence (See Appendix 3).

## 2.1 Evaluation

We use two complementary evaluations that capture different aspects of grammatical behavior.

Minimal-pair benchmarks: we use BLiMP (Warstadt et al., 2020) and its CHILDES-adapted version Zorro (Huebner et al., 2021). They test whether a model assigns higher probability to a grammatical sentence than to a minimally differing ungrammatical one.

Grammaticality of generated utterances Free generations were scored using Child Conversational Grammaticality (ChildCG, Nikolaus et al., 2024), a model adapted to CHILDES and its conversational context. In addition, we used a measure based on an off-the-shelf Grammatical Error Correction model (e.g., Rothe et al., 2021). For each run and generation seed, we sampled up to 10,000 utterances using the Beginning-Of-Sentence (BOS) token as a prompt.

## 3 Results

First, we compared the baseline models, trained only on linguistic input, with the same models further fine-tuned using the grammar topline reward model (Figure 2 in Appendix). The topline improves both minimal-pair benchmark performance and the grammaticality of generation, showing the current framework can, in principle, lead to grammatical learning above and beyond pretraining.

Moving to the empirical reward models, we found no consistent improvement on the minimalpair benchmarks (Figure 3 in Appendix). Across Zorro and BLiMP, none of the reward types reliably improved over the baseline. This echoes a broader pattern in recent studies showing that interactive strategies struggle to improve performance on these benchmarks (see also Discussion). In contrast, the generation results showed measurable effects on grammaticality (Figure 1). Across both ChildCG and GEC, the overall trends were largely consistent. Structural alignment produced the strongest and most consistent improvement across scales and metrics. Communicative feedback showed an overall positive effect, especially under ChildCG, but the gains were smaller. In contrast, semantic contingency and affective feedback generally reduced grammaticality below the baseline.

Regarding the effect of pretraining size, the lowresource condition at 0.1M was visibly noisier, with larger variability across runs, excessively high nonword rates (Figure 4 in Appendix, bottom row), and overall low grammaticality scores. The effects became much more consistent from 1M words onward, suggesting that a minimum amount of pretraining data is needed before reward-specific effects can emerge robustly.

We tested these qualitative patterns statistically using a mixed-effects logistic model. Predictors included reward type, pretraining scale, and their interaction. We additionally added the following predictors as controls: a) utterance length, since longer utterances may naturally provide more opportunities for errors, and b) rate of non-words generated, since out-of-dictionary forms may be penalized by grammaticality classifiers for reasons that are not strictly grammatical. Finally, we included a random intercept for generation run to account for the fact that sampled utterances from the same seeds are not independent.

Table 1 reports model-estimated changes in grammaticality probability relative to the baseline, using the more robust pretraining scales (1M and 10M). Mirroring the pattern in Figure 1, structural alignment showed the clearest improvement. In contrast, semantic contingency and affective feedback both reduced grammaticality. Communicative feedback improved grammaticality at 1M and showed a weaker effect at 10M. When communicative feedback was split into its two components, only clarification requests showed a significant effect, whereas the acknowledgment-based reward model did not.

![](images/7a3bb00bb1684afa9cd17e4b57821486f3ee8ec76fe9b37d0ac9f2f60d64c36d.jpg)

Figure 1: Generated-utterance grammaticality across pretraining scales for each reward. The top row shows results from the CHILDES-adapted ChildCG classifier, while the bottom row shows results using a standard GEC-based metric. Within each panel, baseline and topline trajectories are repeated to make the reward trajectory directly comparable. Means are computed over training-seed units after averaging over generation seeds. Error bars show ±1 standard deviation across training seeds.
<table><tr><td>Reward family</td><td>1M</td><td>10M</td></tr><tr><td>Structural alignment</td><td> $+ . 0 7 7 ^ { * * * }$ </td><td>+.060***</td></tr><tr><td>Semantic contingency</td><td>-.092***</td><td>−.061***</td></tr><tr><td>Affective feedback</td><td>-.038*</td><td>-.031*</td></tr><tr><td>Communicative feedback</td><td>+.033*</td><td>+.024†</td></tr><tr><td>Acknowledgment</td><td>+.017</td><td>+.017</td></tr><tr><td>Clarification request</td><td>+.050***</td><td>+.031*</td></tr></table>

Table 1: Model-estimated change in grammaticality probability relative to baseline, using the ChildCG metric at the 1M and 10M pretraining scales. $\dag p < . 1 0$ $^ { * } p < . 0 5 , ^ { * * } p < . 0 1 , ^ { * * * } p < . 0 0 1$

## 4 Discussion

We investigated which forms of caregiver feedback support grammaticality within an RLHF-inspired framework. We isolated and compared major feedback types, providing insight into their specific effects.

Our main finding is that structural alignment systematically improves grammaticality. Previous corpus work found that caregivers’ alignment patterns predicted children’s later language outcomes, with structural alignment specifically predicting increase in a measure of grammatical knowledge (Fusaroli et al., 2023). Complementing this line of work, the current result demonstrates, at scale, a plausible mechanistic account: Caregivers’ structural alignment can support grammar learning because this reinforces the child’s use of structures that are grammatical. This need not involve deliberate teaching: As competent speakers, caregivers are naturally more likely to re-use parts of children’s utterances when they are well-formed than when they are ill-formed.

Regarding communicative feedback, it also improved grammaticality, though to a lesser extent. The improvement is consistent with prior work, especially on clarification requests (Nikolaus and Fourtassi, 2026; Clark, 2020).

For affective feedback and semantic contingency, neither improved grammaticality; instead, both reduced it. Echoing our finding, previous research on parenting behavior did not find a specific relation between affective behavior and language development (Rivero et al., 2023). The authors suggested that, on its own, this type of feedback may be less directly related to language learning than to sociocognitive development. Our supplementary analyses nevertheless suggest that affective feedback may still support some aspects of language development, though not necessarily in grammar. For instance, results in Figure 4 and model-generated samples (Table 2) show that affective feedback increased utterance length relative to the baseline. This suggests that it may encourage longer vocalizations, which—while increasing opportunities for grammatical errors—can potentially support early phonological development (Goldstein et al., 2003; Nikolaus et al., 2022; Warlaumont et al., 2014).

For semantic contingency, supplementary analyses (Figure 4) show increased content-word use relative to baseline and greater lexical diversity than the other feedback types. These effects are consistent with previous studies linking caregivers semantic and lexical contingency to children’s subsequent lexical growth and diversity (e.g., Fusaroli et al., 2023; Che et al., 2018).

As for why semantic contingency reduced grammaticality, one possible explanation is that our measure captured two competing mechanisms. Our original hypothesis was that high semantic contingency signals communicative success, which should be more likely when children produce wellformed utterances that caregivers can readily understand (Nikolaus and Fourtassi, 2023). However, high semantic contingency can also arise when caregivers reformulate children’s errors (Chouinard and Clark, 2003). Ungrammatical utterances may therefore elicit semantically contingent responses, as caregivers reformulate the incorrect form while preserving the intended meaning. A reward model favoring such responses could inadvertently reinforce ungrammatical productions. Future work should disentangle these mechanisms.

Thus, different types of feedback push generation in different directions: some improve grammaticality, while others encourage other properties of language use. Future work should investigate how these feedback types may support complementary aspects of language learning beyond grammar.

## Minimal-Pair vs. Generation-Based Evaluation

The topline reward improved on baseline in grammatical benchmarks, indicating the viability of this learning framework (Figure 2). However, the effects of the empirical rewards were observed in model generation (Figure 1), but not in minimalpair benchmarks (Figure 3). This echoes a broader pattern in recent studies showing that interactive strategies generally struggle to improve performance on minimal-pair benchmarks of grammar such as BLiMP and Zorro (Zhao et al., 2023; Padovani et al., 2025; Nikolaus and Fourtassi, 2026; Stöpler et al., 2025) and work showing that child linguistic environment has more apparent effects in production-based evaluation than on minimal-pair benchmarks (Bunzeck and Zarrieß, 2026).

This raises an important question: To what extent can changes in grammatical production be taken as evidence of changes in underlying grammatical knowledge? This echoes a long-standing debate in child language research (e.g., Tomasello, 2000; Fisher, 2002)

In the present study, we specifically found that learning from caregiver-like reward changes the model’s generative behavior. The generation-based results are informative for theories of language development. Unlike corpus-level correlations, the rewards here were used as interventions within a learning system, allowing us to test whether optimizing for these signals causally shifts the model’s production policy in grammar-relevant directions.

That said, a model becoming more grammatical in free generation does not necessarily mean that it has learned new grammatical regularities— or has unlearned ungrammatical patterns—in a way that generalizes beyond the specific utterance types rewarded during fine-tuning. Minimalpair benchmarks—though not perfect (Martínez et al., 2023; Weissweiler et al., 2025)—are useful precisely because they provide out-of-distribution tests for general learning. Our null results on BLiMP/Zorro (Figure 3) therefore matter. They suggest that the empirical rewards did not produce grammatical generalization—at least, not of the kind captured by these benchmarks.

Although this interpretation is plausible, it would be premature to draw it here without further research. Existing benchmarks do not exhaust the space of possible grammatical generalizations. Even child-adapted benchmarks such as Zorro primarily target classic grammatical phenomena, which overlap only partially with errors common in child production (Saxton et al., 2005b; Hiller and Fernandez, 2016; Nikolaus et al., 2024). This is relevant because in RL, errors in productions are the ones that receive direct pressure from the reward.

Testing this learning claim, thus, remains a direction for future work. It requires conducting an exploration of the grammatical errors found in both child and model productions, and using these bottom-up insights to build targeted tests around the relevant phenomena.

## Limitations

A limitation of this work is that we captured only the verbal component, which could have underestimated the learning challenge associated with credit assignment (Pinker, 1979; Marcus, 1993). In reality, children must interpret feedback in real time within a multimodal context (Behne et al., 2012; Adamson et al., 2004; Fourtassi and Frank, 2020). This makes credit assignment especially challenging, because the learner must disambiguate among several possible sources of error beyond the combinatorial structure of the child’s utterance. Valenced feedback may reflect not only how symbols are combined, but also a mispronounced word or an inappropriate lexical choice given the visual context.

At the same time, restricting our analysis to verbally expressed caregiver feedback may have underestimated the richness of the feedback available to children. Caregivers can signal whether and how they have understood a child through multiple channels, including linguistic responses, prosody, gaze, gesture, and facial expression.

A second limitation of this work is that we used a single GPT-2-style architecture. Although we used a fully crossed design over pretraining, RLHF fine-tuning, and generation seeds to account for multiple sources of variability, we did not establish generality across model architectures, mainly because duplicating this large set of experiments across several architectures would require substantial computational resources.

This limitation, howevever, should be interpreted in light of our objective. We do not propose a general-purpose method for improving language models; rather, we use a controlled child-like neural learner as an experimental testbed to examine whether naturally occurring caregiver feedback contains information that can support grammatical development. Nevertheless, replication across architectures would strengthen the generality of this conclusion.

Another potential concern is that reward finetuning could narrow the model’s production distribution, reducing diversity and potentially harming generalization. Our implementation incorporates several safeguards against an excessive loss of diversity, including entropy and language-modeling regularization, adaptive KL control to limit divergence from the pretrained distribution, and randomly sampled short prompts that encourage exploration across a broad range of constructions during the fine-tuning stage (see Appendix 5).

However, some narrowing of the production distribution is both expected and consistent with learning if the model shifts away from ill-formed constructions, effectively “unlearning” ungrammatical options—the primary hypothesized role of negative evidence in grammar (Saxton, 1997).

For instance, in our analyses, the grammar topline was relatively the least diverse condition (Figure 4, Lexical Entropy, Appendix), yet it produced the clearest gains on the minimal-pair benchmarks (Figure 2, Appendix). Thus, reduced diversity need not preclude grammatical generalization and may sometimes accompany it.

Finally, beyond final performance, interactive feedback may also shape learning trajectories. For example, Ma et al. (2024) show that interaction can reduce the amount of data or experience required to reach comparable performance in word learning. Nikolaus and Fourtassi (2021) found that feedbackbased models align with the developmental order of several aspects of visually grounded grammar learning. A direction for future work is to investigate these dynamics within a framework such as the one outlined here, in which realistic caregiver feedback serves as a reward signal.

## Ethical considerations

This work uses existing child-caregiver corpora and does not involve new data collection. Because the data involve children, we treat the corpora as sensitive and report only aggregate model-level results, without attempting to identify individual children, caregivers, or families. The proposed modeling framework is intended as a tool for studying hypotheses about language development, not for evaluating individual children or caregivers.

## Acknowledgments

This work was supported by the ANR MACOMIC project (ANR-21-CE28-0005-01) and was carried out within the Institut Convergence ILCB (ANR-16-CONV-0002). This work was performed using HPC and storage resources from GENCI–IDRIS (Grant 2025-AD011013886R3) on the Jean Zay supercomputer.

This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 945304 – Cofund AI4theSciences hosted by PSL University.

## References

Lauren B Adamson, Roger Bakeman, and Deborah F Deckner. 2004. The development of symbol-infused joint engagement. Child development, 75(4):1171– 1187.

Abhishek Agrawal, Benoît Favre, and Abdellah Fourtassi. 2026. Scaffolding early dialogue: A unified account of response contingency in child–caregiver interaction. Cognitive Science, 50(7):e70245.

Abhishek Agrawal, Mitja Nikolaus, Benoit Favre, and Abdellah Fourtassi. 2024. Automatic Coding of Contingency in Child-Caregiver Conversations. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 1856– 1870, Torino, Italia.

Tanya Behne, Ulf Liszkowski, Malinda Carpenter, and Michael Tomasello. 2012. Twelve-month-olds’ comprehension and production of pointing. British Journal ofDevelopmental Psychology, 30(3):359–375.

Roger Brown and Camille Hanlon. 1970. Derivational complexity and order of acquisition in child speech. Cognition and the development of language.

Bastian Bunzeck and Sina Zarrieß. 2026. Child-directed speech facilitates production, not comprehension, in babylms. In Proceedings of the 30th Conference on Computational Natural Language Learning, pages 227–249.

Elizabeth S Che, Patricia J Brooks, Maria F Alarcon, Francis D Yannaco, and Seamus Donnelly. 2018. Assessing the impact of conversational overlap in content on child language growth. Journal of Child Language, 45(1):72–96.

Michelle M. Chouinard and Eve V. Clark. 2003. Adult reformulations of child errors as negative evidence. Journal ofChild Language, 30(3):637–669.

Eve V. Clark. 2020. Conversational Repair and the Acquisition of Language. Discourse Processes, 57(5- 6):441–459.

Rick Dale and Michael J. Spivey. 2006. Unraveling the Dyad: Using Recurrence Analysis to Explore Patterns of Syntactic Coordination Between Children and Caregivers in Conversation. Language Learning, 56(3):391–430.

M. J. Demetras, Kathryn Nolan Post, and Catherine E. Snow. 1986. Feedback to first language learners: the role of repetitions and clarification questions\*. Journal ofChild Language, 13(2):275–292.

Raquel Fernandez and Robert M Grimm. 2014. Quantifying Categorical and Conceptual Convergence in Child-Adult Dialogue. In Proceedings of the Annual Meeting ofthe Cognitive Science Society, volume 36, page 6.

Cynthia Fisher. 2002. The role of abstract syntactic knowledge in language acquisition: a reply to. Cognition, 82(3):259–278.

Abdellah Fourtassi and Michael C Frank. 2020. How optimal is word recognition under multimodal uncertainty? Cognition, 199:104092.

Ruthe Foushee, Dan Byrne, Marisa Casillas, and Susan Goldin-Meadow. 2022. Getting to the root of linguistic alignment: Testing the predictions of interactive alignment across developmental and biological variation in language skill. In Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, volume 44.

Riccardo Fusaroli, Ethan Weed, Roberta Rocca, Deborah Fein, and Letitia Naigles. 2023. Caregiver linguistic alignment to autistic and typically developing children: A natural language processing approach illuminates the interactive components of language development. Cognition.

Michael H Goldstein, Andrew P King, and Meredith J West. 2003. Social interaction shapes babbling: Testing parallels between birdsong and speech. Proceedings of the National Academy of Sciences, 100(13):8030–8035.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2022. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. In Proceedings of the International Conference on Learning Representations.

Sarah Hiller and Raquel Fernandez. 2016. A Datadriven Investigation of Corrective Feedback on Subject Omission Errors in First Language Acquisition. In Proceedings ofThe 20th SIGNLL Conference on Computational Natural Language Learning, pages 105–114, Berlin, Germany. Association for Computational Linguistics.

Kathy Hirsh-Pasek, Lauren B Adamson, Roger Bakeman, Margaret Tresch Owen, Roberta Michnick Golinkoff, Amy Pace, Paula KS Yust, and Katharine Suma. 2015. The contribution of early communication quality to low-income children’s language success. Psychological science, 26(7):1071–1083.

Kathy Hirsh-Pasek, Rebecca Treiman, and Maita Schneiderman. 1984. Brown & Hanlon revisited: mothers’ sensitivity to ungrammatical forms. Journal ofChild Language, 11(1):81–88.

Erika Hoff-Ginsberg. 1987. Topic relations in motherchild conversation. First Language, 7(20):145–158.

Philip A. Huebner, Elior Sulem, Fisher Cynthia, and Dan Roth. 2021. BabyBERTa: Learning More Grammar With Small-Scale Child-Directed Language. In Proceedings of the 25th Conference on Computational Natural Language Learning, pages 624–646, Online. Association for Computational Linguistics.

Ziqiao Ma, Zekun Wang, and Joyce Chai. 2024. Babysit A Language Model From Scratch: Interactive Language Learning by Trials and Demonstrations. In ICML 2024 Workshop on LLMs and Cognition.

Brian MacWhinney. 2000. The childes project. Computational Linguistics, 26(4):657–657.

Gary F. Marcus. 1993. Negative evidence in language acquisition. Cognition, 46(1):53–85.

Héctor Javier Vázquez Martínez, Annika Heuser, Charles Yang, and Jordan Kodner. 2023. Evaluating neural language models as cognitive models of language acquisition. In Proceedings ofthe 1st Gen-Bench Workshop on (Benchmarking) Generalisation in NLP, pages 48–64.

Lillian R. Masek, Brianna T. M. McMillan, Sarah J. Paterson, Catherine S. Tamis-LeMonda, Roberta Michnick Golinkoff, and Kathy Hirsh-Pasek. 2021. Where language meets attention: How contingent interactions promote learning. Developmental Review, 60:100961.

Jonas Mayer Martins, Ali Hamza Bashir, Muhammad Rehan Khalid, and Lisa Beinborn. 2025. Once upon a time: Interactive learning for storytelling with small language models. In Proceedings ofthe First BabyLM Workshop, pages 454–468, Suzhou, China. Association for Computational Linguistics.

Thomas Misiek, Benoit Favre, and Abdellah Fourtassi. 2020. Development of Multi-level Linguistic Alignment in Child-adult Conversations. In Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, pages 54–58, Online. Association for Computational Linguistics.

Kanishka Misra and Kyle Mahowald. 2024. Language models learn rare phenomena from less rare phenomena: The case of the missing aanns. In Proceedings ofthe 2024 conference on empirical methods in natural language processing, pages 913–929.

Elissa L. Newport, Henry Gleitman, and Lila R. Gleitman. 1977. Mother, I’d Rather Do It Myself: Some Effects and Non-Effects of Maternal Speech Style. In Sentence First, Arguments Afterward. Oxford University Press, New York.

M. Nikolaus and A. Fourtassi. 2026. Modelling Children’s Language Learning via Caregiver Feedback in Natural Conversations. Philosophical Transactions of the Royal Society B: Biological Sciences, 381.

Mitja Nikolaus, Abhishek Agrawal, Petros Kaklamanis, Alex Warstadt, and Abdellah Fourtassi. 2024. Automatic Annotation of Grammaticality in Child-Caregiver Conversations. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 1832–1844, Torino, Italia.

Mitja Nikolaus and Abdellah Fourtassi. 2021. Modeling the Interaction Between Perception-Based and Production-Based Learning in Children’s Early Acquisition of Semantic Knowledge. In Proceedings of the 25th Conference on Computational Natural Language Learning (CoNLL), pages 391–407.

Mitja Nikolaus and Abdellah Fourtassi. 2023. Communicative Feedback in Language Acquisition. New Ideas in Psychology.

Mitja Nikolaus, Laurent Prévot, and Abdellah Fourtassi. 2022. Communicative Feedback as a Mechanism Supporting the Production of Intelligible Speech in Early Childhood. In Proceedings ofthe 44th Annual Meeting of the Cognitive Science Society.

Mitja Nikolaus, Laurent Prévot, and Abdellah Fourtassi. 2023. Communicative Feedback in Response to Children’s Grammatical Errors. In Proceedings for the 45th Annual Meeting of the Cognitive Science Society.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Francesca Padovani, Bastian Bunzeck, Manar Ali, Omar Momen, Arianna Bisazza, Hendrik Buschmeier, and Sina Zarrieß. 2025. Dialogue is not enough to make a communicative babylm (but neither is developmentally inspired reinforcement learning). In Proceedings ofthe First BabyLM Workshop, pages 421–435.

Steven Pinker. 1979. Formal models of language learning. Cognition, 7(3):217–283.

Steven Pinker. 1989. Learnability and Cognition: The Acquisition of Argument Structure. MIT press.

Magda Rivero, Rosa Vilaseca, María-José Cantero, Clara Valls-Vidal, and David Leiva. 2023. Relations between positive parenting behavior during play and child language development at early ages. Children, 10(3):505.

Sascha Rothe, Jonathan Mallinson, Eric Malmi, Sebastian Krause, and Aliaksei Severyn. 2021. A Simple Recipe for Multilingual Grammatical Error Correction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 702–707, Online. Association for Computational Linguistics.

Matthew Saxton. 1997. The Contrast Theory of negative input. Journal ofChild Language, 24(1):139–161.

Matthew Saxton, Phillip Backley, and Clare Gallaway. 2005a. Negative input for grammatical errors: effects after a lag of 12 weeks. Journal of Child Language, 32(3):643–672.

Matthew Saxton, Carmel Houston–Price, and Natasha Dawson. 2005b. The prompt hypothesis: Clarification requests as corrective input for grammatical errors. Applied Psycholinguistics, 26(3):393–414.

Ted Schoneberger. 2010. Three myths from the language acquisition literature. The Analysis of Verbal Behavior, 26(1):107–131.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv preprint.

Jeffrey L. Sokolov and Catherine E. Snow. 1994. The changing role of negative evidence in theories of language development. In Input and interaction in language acquisition, pages 38–55. Cambridge University Press, New York, NY, US.

Lennart Stöpler, Rufat Asadli, Mitja Nikolaus, Ryan Cotterell, and Alex Warstadt. 2025. Towards developmentally plausible rewards: Communicative success as a learning signal for interactive language models. arXiv preprint arXiv:2505.05970.

Michael Tomasello. 2000. Do young children have adult syntactic competence? Cognition, 74(3):209–253.

Anne S. Warlaumont, Jeffrey A. Richards, Jill Gilkerson, and D. Kimbrough Oller. 2014. A Social Feedback Loop for Speech Development and Its Reduction in Autism. Psychological Science, 25(7):1314–1324.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. Blimp: The benchmark of linguistic minimal pairs for english. Transactions ofthe Associationfor Computational Linguistics, 8:377–392.

Leonie Weissweiler, Kyle Mahowald, and Adele Goldberg. 2025. Linguistic generalizations are not rules: Impacts on evaluation of lms. In Proceedings of the Second International Workshop on Construction Grammars and NLP, pages 61–74.

G. J. Whitehurst and M. C. Valdez-Menchaca. 1988. What is the role of reinforcement in early language acquisition? Child Development, 59(2):430–440.

Qihui Xu, Robert Ralston, Madison Meares, and Vladimir Sloutsky. 2025. Wanting to be understood: Modeling interaction in early language learning. In Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, volume 47.

Xingmeng Zhao, Tongnian Wang, Sheri Osborn, and Anthony Rios. 2023. BabyStories: Can reinforcement learning teach baby language models to write better stories? In Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, pages 186–197, Singapore. Association for Computational Linguistics.

## Appendix 1: Supplementary Analyses

Benchmarks: minimal−pair accuracy  
![](images/1bd4e512c0c7c9b022e2deb84475bd63cd0d60d5b8030ae21492c7de717e60b7.jpg)

Generated utterances: grammaticality rate  
![](images/56be579212667a0e2289106830b65113e194ecf0572cf2d9b72fd477fd38c914.jpg)  
Figure 2: Baseline and Topline performance across pretraining scales. The top row reports minimal-pair benchmark accuracy for Zorro and BLiMP. The bottom row reports grammaticality of generated utterances, using the ChildCG or GEC-based metric. Points show means across training seeds (after averaging over generation seeds). Error bars show ±1 standard deviation across these independent units.

![](images/d4f39d7a8c889b237dbfcd4d74ce3b18e806ba2e272796fb9cb47bce36ba277f.jpg)

Figure 3: Benchmark performance across pretraining scales for each reward family. Rows show Zorro and BLiMP minimal-pair accuracy; columns show the four empirical reward families. Within each panel, the baseline and topline trajectories are repeated to make the reward trajectory directly comparable at each scale. Means are computed over training seeds, after averaging over generation seeds. Error bars show ±1 standard deviation across independent units.  
![](images/d99cbcf748a18f67beb3170b568a64a52a2bc3067baef862bedc26ce445910be.jpg)  
Figure 4: Properties of generated utterances: Sentence length, pooled lexical entropy, function-word density, and rate of non-words. Points show means for each reward across training seeds, with error bars indicating ±1 standard deviation across these seeds. Within each panel, baseline and topline trajectories are repeated to make the reward trajectory directly comparable at each scale.

Appendix 2: Examples of Model-Generated Utterances
<table><tr><td>Reward type</td><td>ChildCG = 1 (grammatical)</td><td>ChildCG = 0 (ungrammatical)</td></tr><tr><td>Baseline</td><td>you want to look at some more pictures? oh, is that what it looks like? it is a big boy now isn&#x27;t it? have you not any other little boys?</td><td>this piece of cake? because you can see mummy&#x27;s poorly sick. the dog is saying. he is the name of the little girl.</td></tr><tr><td>Topline</td><td>what are you doing? there is a horse. that is a girl. are you going back? that is a chair. what is that one? i will put it there.</td><td>why is that side?</td></tr><tr><td colspan="2">Alignment</td><td>so that is a duck. he is a turtle. you are a good boy. i am a monkey.</td></tr><tr><td colspan="2">Contingency</td><td>where is baby sheep? what is big truck? he a lion. yes a banana puppy. a big flower red.</td></tr><tr><td colspan="2">feedback</td><td>and then i will put it in it. if i do it and then we can do it and put it in? i will put it over here if i put it in there. now i will have to clean this with your room won&#x27;t you? we have to wash her hair first okay? want mama do it? we will shall we put it all together? because your teeth i am watching this picture aren&#x27;t they?</td></tr><tr><td rowspan="3">Communicative feedback</td><td>Acknowledgment they are called flowers. that is a gold straw.</td><td>have to use your spoon and pretend. you will use them ones with this bits. and that is the baby. all day and then the is warm.</td></tr><tr><td>you have a pink one on there. Clarification i don&#x27;t know. can you see it?</td><td>but two grey stars. look at that isn&#x27;t it?</td></tr><tr><td>request can&#x27;t you?</td><td>look it, there is it? see you shouldn&#x27;t?</td></tr></table>

Table 2: Examples of utterances generated by RL-fine-tuned models at the 10M-word scale (generation was qualitatively similar at the 1M-word scale). Examples are organized by reward types. Unshaded cells show utterances that were classified as grammatical by ChildCG, while shaded cells show utterances classified as ungrammatical.

## Appendix 3: Automatic labeling of rewards

We automatically annotated caregiver feedback signals in child-caregiver conversations and used these annotations to supervise reward model training (see Appendix 4).

## Communicative feedback

Clarification requests Clarification requests were identified using a DeBERTa-v3-xsmall classifier fine-tuned on manually annotated caregiver responses in CHILDES, following prior work (Nikolaus and Fourtassi, 2026).

Acknowledgments To identify caregiver acknowledgments, we relied on the feature-based algorithm described in Nikolaus et al. (2023), which was specifically designed for CHILDES data and combined common keywords (e.g., “okay”, “alright”, and “yeah”) as well as repetition ratios capturing repetition-based acknowledgments (e.g., “It isn’t very nice, is it?” – “It isn’t.”).

## Structural Alignment

Following previous research (Dale and Spivey, 2006; Fernandez and Grimm, 2014), a child structure was defined as a POS bigram. We used spaCy (en\_core\_web\_sm)<sup>1</sup> to tag each child utterance and caregiver response for part-of-speech (POS), excluding punctuation and empty tokens. Caregiver alignment was computed as a normalized score of bigram reuse, corresponding to the size of the intersection between the child and caregiver POSbigram sets, divided by the size of the larger set.

## Semantic Contingency

Following previous research (e.g., Fusaroli et al., 2023; Foushee et al., 2022), semantic contingency was computed using sentence embedding similarity. Here, we used all-MiniLM-L6-v2<sup>2</sup> to encode the child utterance and caregiver response. We computed the cosine similarity between the two embeddings, and clipped the resulting score to the [0, 1] range.

## Affective feedback

We annotated affective properties of caregiver responses using roberta-base-go\_emotions.<sup>3</sup> The model was applied to adult responses only.

For each caregiver utterance, the classifier returned a probability distribution over several labels. We used the probabilities for emotions related to supportiveness, approval, and warmth.

## Appendix 4: Baseline and Reward model training

## Baseline models

For the input-based baselines, we trained a small GPT-2-style causal language model on caregiver utterances only. The model used 2 hidden layers, 8 attention heads, and a hidden size of 512. We trained a word-level tokenizer on the training data, retaining words that appeared at least twice and capping the vocabulary at 5,000 types. Models were optimized with the standard GPT-2 causal language-modeling objective using AdamW and a batch size of 256. For each data scale, we held out 10% of the data for validation and used validation loss for early stopping and checkpoint selection. To assess the effect of input size, we trained models on three amounts of caregiver language: 0.1M, 1M, and 10M words. Each condition was run with three different random seeds.

## Reward models

For each of the four caregiver feedback dimensions above, we trained a reward model that maps a child utterance to the corresponding valence of the caregiver response, as annotated using the procedures described in Appendix 3.

All reward models share the same pretrained model: microsoft/deberta-v3-xsmall (He et al., 2022), which we fine-tuned using a linear regression layer and a mean squared error (MSE) loss to predict the target reward for a given utterance.

All reward models were trained on the CHILDES child–caregiver conversation pairs (one row per child utterance with the immediately following caregiver response), the same pre-processed pipeline above. We split the data into 90% training and 10% held-out evaluation with a fixed random seed. We used the AdamW optimizer with an initial learning rate of $1 . 4 1 \times 1 0 ^ { - 5 }$ , a batch size of 128, and early stopping based on MSE on the held-out validation set. The best checkpoint is selected on held-out MSE and used as the frozen reward model during PPO fine-tuning.

## Topline Reward Model

The topline reward model was obtained using the same procedure, except that instead of learning the mapping between child utterances and the feedback valence, it was trained on the controlled minimal-pair datasets used in Zorro and BLiMP. More specifically, the model was trained to classify sentences from these datasets according to their labels (i.e., grammatical or ungrammatical).

## Appendix 5: RL Fine-tuning Details

After pretraining on child-directed CHILDES utterances, we fine-tune each language model with Proximal Policy Optimization (PPO; Schulman et al., 2017), implemented using Hugging Face’s TRL library. For each fine-tuning step, we (a) sample utterances from the current language model, (b) compute the corresponding rewards using the reward model, and (c) update the model weights using PPO.

To obtain a diverse set of generated utterances, we randomly prompted the model with short utterance prefixes (the first one or two tokens) sampled from the pretraining data, namely the caregiver utterances used to train the input-based baseline model. We additionally included an entropy regularization term (0.001) in the loss.

The models were fine-tuned for a maximum of 6,000 steps, and the best checkpoint was selected based on mean reward. To discourage excessively short or long utterances, we applied rejection sampling by assigning a reward of −1 to generated utterances that were shorter than three tokens or failed to produce an end-of-sequence token within 20 tokens.

To mitigate language drift, a known issue in Reinforcement Learning (RL) fine-tuning, we also added a small language-modeling regularization term $( \mathrm { w e i g h t } = 0 . 0 0 1 )$ . We used the TRL default optimizer and learning rate. Other hyperparameters are shown in Table 3. All other PPO hyperparameters were not changed from the default values as implemented in the Huggingface TRL library.

Each PPO run uses one NVIDIA H100 GPU (96 GB) with bf16 mixed precision and completes in approximately 6 GPU-hours.

<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Max PPO total steps</td><td>6000</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Mini-batch size</td><td>512</td></tr><tr><td>KL controller</td><td>adaptive, target  $\mathrm { K L } ^ { \star } = 6$ </td></tr><tr><td>Entropy coef.  $c _ { e }$ </td><td> $1 \times \mathrm { 1 0 ^ { - 3 } }$ </td></tr><tr><td>LM mixing coef.  $\lambda _ { \mathrm { L M } }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Score clipping</td><td>disabled</td></tr><tr><td>Sampling</td><td> $T { = } 1 . 0 , \mathrm { t o p } { - } p { = } 1 . 0 , \mathrm { t o p } { - } k { = } 0$ </td></tr><tr><td>Precision</td><td>bf16 mixed</td></tr><tr><td>Eval frequency</td><td>every 100 steps</td></tr><tr><td>Log frequency</td><td>every 10 steps</td></tr><tr><td>Early stopping</td><td>no reward gain for 10 log windows</td></tr></table>

Table 3: PPO fine-tuning hyper-parameters, shared across the four caregiver-feedback rewards, the grammar topline, and all pretraining scales.