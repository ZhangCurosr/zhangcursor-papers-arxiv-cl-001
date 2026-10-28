# Specialization, Diversity, and Scaling in Child-Scale Language Models

Steven Y. Feng, Michael C. Frank Stanford University {syfeng,mcfrank}@stanford.edu

## Abstract

Modern language models are trained on far more language than children experience, but we know much less about learning from small, naturalistic datasets. We train small GPT-2 models on transcripts of speech with children from dense home recordings (BabyView) and on synthetic child-directed dialogue (TinyDialogues), varying the amount, source, and diversity of training data. Models trained on one child's language environment predict that environment substantially better than other children's environments. Much of this specialization is lexical and distributional: a simple 5- gram model reproduces the cross-child pattern (r = .91). Training on data from multiple children clearly improves prediction of pooled heldout language relative to training on a single child's input without a similarly clear improvement on grammatical benchmarks. Across data sizes, naturalistic and synthetic training inputs also produce different grammatical and lexicalsemantic patterns, with important differences between grammatical phenomena. Together, these results show that small-data language learning depends not only on how much language a model sees, but on which environments it comes from and how well that input matches what the model is asked to predict.

## 1 Introduction

Modern language models learn from enormous amounts of text. Human children, by comparison, are exposed to far less language during early development: roughly $1 0 ^ { 7 } – 1 0 ^ { 8 }$ words, compared with around $1 0 ^ { 1 2 } \mathrm { - } 1 0 ^ { \mathrm { i } 3 }$ words used to train many current large language models (LLM) (Frank, 2023; Warstadt et al., 2023). This large “data gap" has motivated developmental-scale language modeling, including the BabyLM challenge (Warstadt et al., 2023; Hu et al., 2024). BabyLM keeps the training data fixed and compares different learning methods. We take the complementary approach: we mostly keep the learner fixed and compare different kinds of training data. These models are not meant to reproduce how children learn. We instead use them to ask what properties of small language datasets make learning and generalization easier or harder.

This data gap is not only about the number of words a learner receives. Two learners can see the same number of words but experience very different kinds of language. Children's environments differ in vocabulary, topics, speakers, routines, and linguistic structures. These differences may matter most when training data are limited. For example, a model may perform well on a test set simply because the same words and constructions appeared frequently in its training data. We use distributional support and match to refer to these ideas: whether the language being evaluated is represented in the training data, and how similar the training and evaluation distributions are.

Naturalistic developmental data make it possible to study these questions directly. The BabyView dataset contains egocentric recordings from children's everyday environments, along with natural language transcripts (Long et al., 2024). Since many children were recorded using the same collection procedure, we can train separate models on different children's environments while keeping the broader data-collection setup consistent. Because densely recorded language from individual children is limited, however, our training sets are still smaller than children's estimated total exposure. Thus, we pair BabyView with TinyDialogues (Feng et al., 2024), a synthetic child-directed dialogue corpus that can be generated at larger scales.

We also compare our neural models with a simple count-based baseline. When training data are very small, strong prediction may not require complex neural representations: vocabulary overlap and short-range word patterns may already explain much of the performance. We therefore train a Kneser-Ney 5-gram model on each training set. This baseline serves two purposes. First, it tells us when a simple count model can match or outperform GPT-2 at our data scale. Second, it helps us ask whether effects that look like sophisticated environmental learning can instead be explained by relatively simple distributional statistics.

![](images/204756c940985d2d70dbf6d8b186346d6e0acc16ce7636dee69c6ab4544ca232.jpg)

![](images/eb437d1c7a68140de85e71f418babed334a8e7cd66fb52acf6a1add1e85e1770.jpg)

![](images/9cec05e111cca3b3c7222a3723f427d37e7bc6a6077c068a2d2a793a549005d8.jpg)  
Figure 1: BabyView examples and two central findings. (A) A representative BabyView egocentric frame and a separate excerpt of automatically transcribed child-adult conversation. The egocentric frame and transcript excerpt come from separate BabyView recordings and are presented as illustrative examples. (B) Models specialize to local environments, and a 5-gram reproduces much of the cross-child pattern (r = .91 after excluding same-child comparisons). (C) With 100k training words, more contributing children improve prediction of pooled held-out data; Zorro (grammaticality) shows no clear trend. The dashed diagonal in (B) indicates equality between the adjusted model scores; the horizontal dashed line in (C) indicates theoretical chance performance on Zorro.

With this setup, we investigate three main questions. 1) How strongly do models specialize to individual children's language environments, and what explains that specialization? 2) At a fixed training budget, what changes when the data comes from more children? 3) How do learning curves and benchmark performance differ between natural and synthetic input as the amount of data increases?

Figure 1 highlights the dataset and our two most distinctive findings. Models develop large and persistent advantages for the environments they were trained on, much of which can be explained by lexical and other distributional differences. At a fixed data budget, mixing language from more children improves prediction of pooled held-out language more clearly than grammatical performance. We also examine how natural and synthetic inputs support grammatical and lexical-semantic learning as data increase, using count models as a control.

## 2 Related work

Developmental-scale language modeling. Recent work asks how language models can learn from datasets far smaller than those used in standard pretraining. The BabyLM challenge limits training data and evaluates models on common grammatical and semantic tasks (Warstadt et al., 2023; Hu et al., 2024). Other work trains small LMs on child-directed or developmentally motivated data, including CHILDES-derived corpora (Huebner et al., 2021; Portelance et al., 2023; Yedetore et al., 2023; Hu et al., 2025) and synthetic dialogue (Feng et al., 2024; Zeng et al., 2026). Single-child studies use transcribed input or grounded video and audio (Wang et al., 2023; Vong et al., 2024); Qin et al. (2024) trained on three single-child datasets of roughly 0.2M tokens each. An earlier BabyView study examined scaling, differences among children's input, and developmental vocabulary outcomes on a subset of the current data (Feng et al., 2026). Our study builds on this setting but uses matched data budgets, more training draws and seeds, systematic cross-child comparisons, and additional tests of specialization, persistence, and diversity. Following a controlledrearing approach (Warstadt and Bowman, 2022), we mostly hold the model and training pipeline fixed while varying the input, rather than comparing learning methods on a fixed corpus.

Scaling and data efficiency. Scaling laws describe how LM performance changes as compute, model size, and training data increase. Across large ranges, loss often follows smooth relationships that can be approximated by power laws (Kaplan et al., 2020; Hoffmann et al., 2022; Muennighoff et al., 2023). Related work also examines how to allocate data between pretraining and retrieval (Singh et al., 2026). Most of this evidence, however, comes from datasets many orders of magnitude larger than what we study. At small data scales, factors such as optimization and unseen vocabulary or constructions may matter more. Our goal is to characterize scaling within the child-scale data regime, not to propose or test new scaling laws.

Input distribution, domain match, and variation across children. Models perform better when their training data resembles the data they are evaluated on, and the mixture of training data can affect downstream performance (Gururangan et al., 2020; Longpre et al., 2024; Moore and Lewis, 2010). Developmental research shows that children's language environments differ both in how much language they hear and in what that language contains, including differences in vocabulary and syntax (Rowe, 2012; Rowe and Snow, 2020; Anderson et al., 2021; Masek et al., 2021). These findings motivate separating the amount of input from its composition and its match to the evaluation data.

## 3 Experimental framework

## 3.1 Training corpora

BabyView. We use naturalistic transcripts from the 2026.1 release of the BabyView dataset, which contains egocentric recordings (along with natural language transcripts) of children's everyday environments. After preprocessing and filtering the English data, the available training data contains approximately 4.67M words. Our main scaling series uses nested random subsets of 25k, 50k, 100k, 200k, 400k, 800k, 1.6M, and 3.2M words, with three corpus draws at each data size. We also evaluate the full 4.67M-word corpus with one draw and three model seeds; this is shown separately from the replicated scaling series. For each draw, we randomly order the corpus using a fixed seed, and larger data sizes extend the smaller ones. Our uncertainty intervals therefore describe variation within the observed BabyView data, rather than variation across a hypothetical population of households.

TinyDialogues. As a synthetic comparison, we use child-directed dialogue generated for a twoyear-old child (the age-2 condition) (Feng et al., 2024). Its training sizes range from 25k to 12.8M words. Supplementary experiments vary the target ages of the generated dialogue: M1 combines ages 2, 5, and 10 under the same generation setup, while M2 also changes the generator source. Because M2 changes more than age composition, we do not use it to isolate an age effect (Appendix J).

Individual environments and mixtures. To study specialization, we construct matched 100kword training corpora for 16 children, with three corpus draws and three model seeds per child (144 runs). Each model is evaluated on held-out language from its own child's environment and the other children's environments. Of 18 available children, 16 have enough training data for a 100k-word single-child corpus. To study diversity, we vary the number of contributing children (k ∈ {1, 3, 10, 18}) at a fixed 100k-word total budget, using the 16 eligible children for k = 1 and drawing multi-child mixtures from all 18 children for other values of k. We also perform a 250kword replication as a robustness check. We use 100k words for the primary analysis because this budget allows matched single-child comparisons across 16 eligible children, whereas the 250k-word follow-up only includes six individual children and is thus treated as a secondary replication.

The three experiment families use different model and tokenizer configurations where required; we keep these separate throughout the analyses.

## 3.2 Models and training

The controlled scaling experiments use GPT-2- style causal language models (Radford et al., 2019) at two sizes: 13.9M-parameter micro and 91.6Mparameter small. Each scaling condition uses a separate 7.5k-token BPE tokenizer trained on heldout data from that corpus. Models are trained for 20 epochs over the data, with checkpoints selected using validation loss. The primary checkpoint is the one with the lowest loss on a held-out selection split that is separate from all reported evaluation sets; we also report results from the final checkpoint as a secondary analysis. The individual-environment, persistence, and diversity analyses instead use GPT-2 micro with one shared 16k-token BPE tokenizer across all children (17.2M parameters), and use the final checkpoint for the main analysis.

We match training conditions by the number of sampled word occurrences, not optimization steps or compute. Since every model makes the same number of passes over its dataset, models trained on larger datasets receive more optimization steps. We call these comparisons fixed-recipe scaling and do not interpret them as compute-optimal learning curves. To check whether the low-data comparison with the count model depends on our training setup, we test alternative learning rates, dropout, weight decay, and training lengths from 100k to

1.6M words. For each corpus, model size, and data size, we choose the configuration with the best average validation loss across the three corpus draws, without using the held-out evaluation data, and confirm it with additional model seeds. Stronger regularization and longer training are selected in most cases, suggesting that the fixed recipe is not optimal across this range (Appendix A).

## 3.3 Kneser-Ney 5-gram baseline

For each training corpus, we fit a Kneser-Ney 5- gram (KN-5) model (Kneser and Ney, 1995; Chen and Goodman, 1999) as a simple baseline based on local word statistics. KN-5 lowercases the text and conditions only within utterances, while GPT-2 can use longer context as well as speaker and separator tokens. The two models thus make different assumptions about the data, so we use this comparison as a diagnostic rather than as a perfectly matched architecture ablation.

KN-5 and GPT-2 handle unseen words differently. KN-5 maps all unseen words to a single <unk> symbol, while BPE must assign probability to the full spelling of each word. This can favor KN-5 when many held-out words are unseen during training. We hence base our primary crossover analysis on a vocabulary-aligned comparison, scoring both models on the same in-vocab held-out words. Results using the original closed-vocabulary scoring are reported in Appendix C.

## 3.4 Evaluation

Held-out prediction. We report content bits per word (BPW), where lower values mean better prediction. Our main cross-corpus scaling curves use held-out text from the corresponding corpus. Because the evaluation text therefore differs across corpora, we do not interpret absolute BPW differences between corpora as controlled corpus effects. When direct comparisons are needed, we instead evaluate models on the same text.

Grammatical benchmarks. Zorro (Huebner et al., 2021) contains 46k grammatical minimal pairs covering 13 phenomena. Our primary evaluation compares the summed log-probabilities of the two sentences in each pair. Main figures plot Zorro accuracy. We also report a per-token-normalized Zorro sensitivity analysis and BLiMP (Warstadt et al., 2020) as a supporting benchmark. Because effects differ heavily across phenomena, we refer to these results as “grammatical benchmark performance" rather than treating aggregate accuracy as a direct measure of grammatical competence.

Lexical semantics. We evaluate how well models capture relationships between word meanings using five word-similarity benchmarks (Rubenstein and Goodenough, 1965; Finkelstein et al., 2002; Hill et al., 2015; Gerz et al., 2016; Bruni et al., 2014), in which humans rated how similar or related pairs of words are. For each pair, we compute the cosine similarity between the model's word representations and measure how well they agree with human ratings using Spearman correlation. We extract word representations from the final transformer layer, averaging across subtokens when necessary. This extraction method is fixed rather than choosing the best-performing layer on each test set; absolute scores may differ under other methods (Appendix I.1). Cross-corpus comparisons use identical benchmark pairs. Developmental outcomes are supplementary (Appendix M). We use WordSim in the corpus-scaling comparisons; the fixed-budget diversity experiment instead focuses on held-out prediction and grammaticality.

## 3.5 Statistical analysis

Training runs are replicated across corpus draws and model seeds. Scaling intervals use a two-stage bootstrap over corpus draws and seeds; individualenvironment and persistence analyses use childcluster bootstraps and permutation tests; diversity analyses use HC1 intervals and label permutations. For the low-data scaling analysis, we test whether a standard power law adequately describes the observed curves using criteria specified before examining the results. Full details are in Appendix E

## 4 Individual environments leave strong distributional signatures

## 4.1 Own-environment prediction advantage

We first ask whether models trained on equal amounts of language become specialized to particular children's environments. To test this, we train 144 models (16 children × 3 corpus draws × 3 seeds) on matched 100k-word samples and evaluate each on held-out language from every child's environment. Across these comparisons, models show a strong same-environment advantage (Figure 1B). On average, they achieve 0.77 BPW lower loss on held-out language from the environment they were trained on than language from other children's environments (95% interval: [0.62, 0.92]) All 16 children show this advantage. By contrast,

A Rare and unseen words carry most specialization

![](images/47062970096b15a7b0010302d8f3093c5613d78c869381a2cfc0a4249a122257.jpg)  
B Specialization remains detectable months later

![](images/c0b3c24f1d4e8ba16a3726f44d5e9cccb8f63fe788320ccd462142f1c7e7918a.jpg)  
Figure 2: Specialization is concentrated in rare vocabulary and remains detectable months later. (A) Words unseen in the source child's 100k-word training corpus account for 94.8% of the same-environment advantage while comprising 6.2% of evaluated words. (B) The advantage remains on later recordings but becomes weaker with time.

differences across children on Zorro are small relative to variation from corpus sampling and model runs (variance ratio 0.13). Children's environments thus differ strongly in how well they match each model's training data, without corresponding evidence that some environments provide generally better grammatical input.

## 4.2 Beyond the child's own speech

One possibility is that models specialize mainly by learning the child's own way of speaking. To test this, we first evaluate models using only speech from other people in the child's environment. The same-environment advantage remains essentially unchanged (0.775 BPW versus 0.769 BPW when evaluating all speech). This shows that specialization also extends to the language spoken by other people around the child.

However, these models were still trained on the child's full dialogue. In a separate experiment, we therefore remove the child's own speech from training and compare this with removing the same amount of speech at random. Removing the child's speech specifically leads to worse held-out prediction (+0.26 BPW across 10 children), while Zorro performance is essentially unchanged (Appendix L). Together, these results show that specialization extends beyond predicting the child's own speech, even though that speech also provides useful information during training.

## 4.3 Lexical and distributional differences

The same-environment advantage is highly concentrated in particular words (Figure 2A). About 95% of the effect comes from words that were unseen in the model's own training data, even though these words make up only about 6% of the evaluation set. Cross-child prediction also becomes worse as vocabulary and lexical-distribution mismatch increase. Unseen words can include proper names, household-specific terms, rare objects, and transcription artifacts, and we do not fully separate these sources. We therefore interpret the specialization as mainly lexical and distributional, rather than as evidence that each child's environment produces a different global grammar.

## 4.4 5-gram reproduces cross-child pattern

A simple KN-5 model reproduces much of the same cross-child structure. Even after excluding the 16 same-child comparisons, the KN-5 and neural matrices correlate strongly (r = .91) across the remaining 240 comparisons (Figure 1B; full matrices in Appendix D). KN-5 and GPT-2 hence largely agree about which children's environments are more or less compatible. This suggests that much of the cross-child pattern can be explained by local distributional statistics without requiring a neural explanation. We do not compare the raw KN-5 and GPT-2 effect sizes directly because KN-5 uses a closed vocabulary and the specialization effect is concentrated in unseen words; our main result is the shared pattern across environments.

## 4.5 Specialization persists for months

The specialization effect is not limited to language recorded around the same time as the training data. We train models on each child's input up to a cutoff date and evaluate them on speech from other people in that child's environment at later times. Even

![](images/5c438b505202e6858346eba5f1674b77528cd4bfe33905cd99ad15156bc73521.jpg)

![](images/a2a3f52a7b4f51b4f33e92a12a359cb3e76de7f253b8773aa696d29708e9f652.jpg)  
Figure 3: At a fixed word budget, language from more children improves pooled held-out prediction more clearly than grammatical-benchmark performance. Each point is one 100k-word training dataset (mean across three model seeds); diamonds show the mean for each number of contributing children. (A) Pooled held-out loss decreases as the training set has more children (slope —0.124 BPW per ln k; permutation $p < . 0 0 1 )$ . (B) Zorro shows no clear monotonic trend at the same budget $( p = . 2 8 )$ . The dashed line is theoretical 50% chance on Zorro.

6–12 months later, the mean same-environment advantage remains \~0.46 BPW. The effect becomes weaker over time, decreasing by \~0.04 BPW/month; a separate chronological-prefix analysis shows a similar decline (Figure 2B). Because time and child age increase together, this decline should not be interpreted as a causal effect of time itself. The result shows that environment-specific differences remain detectable months later but become weaker as the child's language environment changes. More details in Appendix F.

## 4.6 Distributional differences & specialization

Finally, we test if differences between environments are sufficient to produce this kind of specialization in a controlled synthetic setting. As we make the synthetic source distributions more different from one another, the same-environment advantage increases steadily. At similar levels of distributional difference, synthetic data can produce specialization of a similar overall size to BabyView, although BabyView's effect is much more concentrated in unseen vocabulary. Distributional differences alone can thus produce substantial specialization, but the synthetic manipulation does not reproduce all of the lexical structure found in the naturalistic data. The amount of distributional difference also does not fully predict the advantage across different synthetic manipulations. Experimental details and comparisons are in Appendix G.

## 5 Diversity improves pooled prediction more clearly than grammar

The specialization results show that training on a particular environment helps a model predict that same environment. We next ask the complementary question: if the total amount of training data stays fixed, what changes when that data comes from more children's environments? We construct 100k-word training sets containing language from 1, 3, 10, or 18 (all available children) environments. As the number of contributing children increases, prediction of the pooled held-out data improves substantially (Figure 3A). BPW decreases by about 0.124 for each unit increase in ln k.

This improvement closely follows how well the training data matches the pooled evaluation data. In a leave-one-dataset-out analysis, unigram Jensen-Shannon divergence (Lin, 1991) explains more variation in performance than the number of contributing children alone $( R ^ { 2 } ~ = ~ . 9 3$ versus .67; Appendix K). In other words, mixtures that more closely resemble the pooled target distribution are easier for the model to predict. We do not interpret this relationship as causal, nor test transfer to a completely new child.

The grammar benchmarks show a different pattern. At 100k words, Zorro shows no clear trend with the number of contributing children $( p \approx . 2 8 )$ and per-token Zorro and BLiMP are likewise null. Thus, mixing more environments clearly improves prediction of pooled held-out language but does not produce a similarly clear grammatical-benchmark improvement at 100k words (Figure 3B).

The 250k-word robustness check is less clearcut: per-token Zorro is positive $( p = . 0 4 4 )$ , while the primary Zorro result is weaker (p = .085) and BLiMP remains null. We therefore do not claim that diversity never helps grammatical generalization (Appendix K). The narrower conclusion is that its benefit for pooled prediction is clear at the primary 100k-word budget, while its effect on grammatical benchmarks is smaller and less consistent.

## 6 Learning across data sizes and sources

## 6.1 Scaling differs by data source

As training data increase, BabyView and age-2 TinyDialogues show different patterns of performance on grammatical and lexical-semantic benchmarks (Figure 4). Across their shared data sizes, BabyView-trained models generally score higher on Zorro, especially at the smallest sizes and again from 1.6M to 3.2M words. The advantage is weaker around 200k-800k words under per-token scoring and on BLiMP. The lexical-semantic comparison points in a different direction: TinyDialogues often performs better on word-similarity benchmarks. Figure 4 evaluates the same benchmark word pairs for both corpora (GPT-2 micro results: Appendix H). As a more restrictive check, the fixed common-word-pair analysis gives a BabyView-minus-age-2 WordSim difference of —0.057 for GPT-2 small across 800k–3.2M words (Appendix I.1). At 25k words, BabyView-trained GPT-2 small scores 59.4% on Zorro versus 48.4% at matched random initialization; the improvement is concentrated in lexical and local-word phenomena, not broad agreement (Appendix I.2). This is a task-specific comparison, not a claim that either corpus supports globally better language learning.

## 6.2 Grammar differences by phenomenon

Overall Zorro accuracy hides large differences among grammatical phenomena. For GPT-2 small across 800k-3.2M words, negative-polarity-item (NPI) licensing accounts for much of the BabyView advantage (about 32 percentage points on that phenomenon; Figure 6 in Appendix C). BabyView also performs better on binding, island effects, and irregular forms, while TinyDialogues does better on several other categories, most consistently quantifiers. These results do not show that BabyView is universally better grammar training data.

The phenomenon-level comparison with KN-5 sharpens this distinction (Appendix C). The count model shows a smaller overall BabyView advantage on Zorro and does not reproduce the neural pattern; most notably, the NPI difference reverses in favor of age-2 TinyDialogues. The observed neural NPI advantage is concentrated in the only/even subset, and differences in licensor frequency do not explain it. We do not have a full mechanistic explanation for this, nor establish a general grammatical advantage for either input source from these results.

## 6.3 Count models stay competitive at low data

As a secondary diagnostic, we compare GPT-2 with KN-5 on held-out prediction. With a fixed 20-epoch training setup and vocabulary-aligned evaluation, GPT-2 small first overtakes KN-5 between 800k and 1.6M words on BabyView and between 200k and 400k on age-2 TinyDialogues. Tuning the neural training setup using validation data moves the BabyView crossover earlier, to 400k– 800k words, but does not change the age-2 smallmodel range. This shows that simple local statistics remain competitive when data are scarce, while the exact crossover depends on training choices (Appendix A; Figure 5).

The fixed-training learning curves improve overall with data, but a single standard power law does not pass our pre-specified fit checks on this lowdata range. This is a limited statement about the tested models and fitting procedure, not a rejection of power-law scaling more generally (Appendix B).

## 7 Discussion

Our results suggest that the gap between LM training data and children's experience is not just about quantity. At small data sizes, it matters which words and constructions are present and how well they match the language being evaluated. Models specialize strongly to individual children's environments, while including more children's environments improves prediction of pooled held-out language without a similarly clear benefit for grammatical benchmarks. These findings highlight how different properties of language input can affect prediction and generalization in different ways.

Although some degree of specialization is expected from differences between training and evaluation distributions, its magnitude and structure in individual children's language environments are empirical questions. We find that these environments leave large and persistent signatures in model predictions, yet much of this specialization is concentrated in a small fraction of evaluated words and reproduced by a simple count model. At the same time, differences among children in grammaticalbenchmark performance are comparatively small, suggesting that environmental match and broader linguistic generalization need not vary together.

![](images/40376dce5339e498485dcb55f03ca52bca4ea64f1398530deb63b3c0b3d96e17.jpg)

![](images/1862ecc194523434c4c29e20cdd2f8a612e53c89361fc5af2215f148409b7352.jpg)  
Figure 4: Scaling of Zorro and WordSim performance (GPT-2 small). (A) Zorro accuracy (50% is chance; color-matched dashed lines show random-initialization baselines: 48.4% for BabyView and 52.0% for age-2 TinyDialogues). (B) Mean WordSim Spearman correlation across five datasets (the same 7.917 word pairs for both corpora). Lines connect the shared 25k-3.2M-word conditions (three corpus draws × three seeds each), with pointwise 95% two-stage bootstrap intervals. Unconnected hollow markers show 4.67M-word results: full BabyView (one draw, three seeds; seed-only range) and size-matched age-2 TinyDialogues (three draws; bootstrap interval). Both use 20 training epochs and validation-selected checkpoints. Appendix H shows GPT-2 micro and larger age-2-only conditions; Appendix J mixed-age comparisons; Appendix I.1 the common-word-pair analysis.

Our scaling results further show that different sources of training data can support different aspects of language learning. BabyView-trained models generally performed better on Zorro, while models trained on synthetic TinyDialogues often performed better on word-similarity benchmarks. The grammatical differences varied substantially across phenomena, and the 5-gram baseline did not reproduce the neural model's pattern, suggesting that simple local word statistics cannot explain all the differences between training sources. These findings are specific to the datasets, models, and benchmarks studied here, and do not establish that either training source is generally better for learning.

Children face the challenge of learning general language patterns from small, highly variable, and often household-specific samples. Yet they still generalize beyond their immediate environments. One possibility is that such learning requires stronger inductive biases or regularization than the models studied here use, together with social, sensory, and multimodal information. Our experiments do not directly test this explanation, but they help identify the distributional differences a theory of child language learning must address.

These findings should not be read as showing that GPT-2 learns language in the same way children do. The value of child-scale data is that it allows us to study realistic variation in language input at a scale where vocabulary overlap and specialization are easy to observe. Future work could test transfer to entirely unseen children, separate names and transcription artifacts from other unfamiliar words, compare larger or compute-matched models, and extend BabyView modeling from transcripts to grounded video and audio.

## 8 Conclusion

At small training-data sizes, language-model performance depends not only on how many words a model sees, but on which environments those words come from. Models specialize strongly to individual children's language environments, especially in vocabulary and other distributional features, and that specialization remains detectable months later. Including more children's environments at a fixed training budget improves prediction of pooled child-language data without a similarly clear gain on grammatical benchmarks. Naturalistic BabyView and synthetic dialogue also support different grammatical and lexical-semantic learning patterns as training data increase; simple count models help identify which patterns can be explained by local statistics.

Together, these findings show why data quantity alone is not enough to describe language learning with small datasets. The content and diversity of the input, and how well it matches the language the model is asked to predict, also matter.

## Limitations

Our conclusions are limited to the models, datasets, and evaluations studied here. BabyView samples only part of children's language and multimodal experience, and dense per-child recordings are scarce: even our pooled 4.67M-word corpus is much smaller than estimates of children's cumulative exposure during early development, which can reach tens of millions of words. Our 100kword individual-child experiments are smaller still. These results should therefore not be interpreted as modeling the full input available to a child. Further, GPT-2-style models are not cognitive models of children, and benchmarks such as Zorro, BLiMP, and WordSim measure particular model behaviors rather than human linguistic knowledge.

More broadly, we use these language models to study the effects of training data on general statistical learners so that these findings can be compared to the performance of children learning language. We do not believe that GPT-2 is a good model of how children themselves learn language, only that a comparison between the two might be informative.

Our scaling comparisons are specific to the training recipes, models, and data sizes tested; the largest synthetic training sets have no matched BabyView counterparts, and TinyDialogues reflects its generation setup. The diversity experiment measures prediction of pooled held-out data from the observed children, not transfer to a new household.

## Ethical Considerations

BabyView contains naturalistic recordings of children and was collected under approved consent and ethics procedures, as described in the original dataset paper (Long et al., 2024). We use the existing dataset for secondary analysis, collect no new participant data, and do not reproduce identifying participant information. The example in Figure 1A is an automatically generated transcript from a separate held-out recording. Further, TinyDialogues is an entirely synthetic dataset. Although our work here is purely basic research, we hope that a better understanding of children's language learning – derived in part from studies comparing children and language models – might eventually lead to better language and speech interventions for children.

## AI Use Statement

We used generative AI to assist with experimental planning and design, implementation, analysis, literature search, and for writing assistance, including rephrasing and improving clarity of author-written text. We reviewed and verified AI-assisted code, results, citations, claims, and text against primary sources and experimental artifacts. The authors directed the research and take responsibility for the final content, including text, claims, or artifacts produced with the assistance of generative AI.

## Acknowledgements

We gratefully acknowledge RunPod, Modal, and NVIDIA's Academic Grant Program for providing the compute resources that enabled this work.

## References

Nina J. Anderson, Susan A. Graham, Heather Prime, Jennifer M. Jenkins, and Sheri Madigan. 2021. Linking Quality and Quantity of Parental Linguistic Input to Child Language Skills: A Meta-Analysis. Child Development, 92(2):484–501.

Elia Bruni, Nam-Khanh Tran, and Marco Baroni. 2014. Multimodal Distributional Semantics. Journal of Artificial Intelligence Research, 49:1–47.

Tyler A. Chang and Benjamin K. Bergen. 2022. Word Acquisition in Neural Language Models. Transactions of the Association for Computational Linguistics, 10:1–16.

Stanley F. Chen and Joshua Goodman. 1999. An empirical study of smoothing techniques for language modeling. Computer Speech & Language, 13(4):359– 394.

Steven Y. Feng, Noah D. Goodman, and Michael C. Frank. 2024. Is Child-Directed Speech Effective Training Data for Language Models? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 22055–22071.

Steven Y. Feng, Alvin W. M. Tan, and Michael C. Frank 2026. Baby Scale: Investigating Models Trained on Individual Children's Language Input. arXiv preprint arXiv:2603.29522.

Larry Fenson, Virginia A. Marchman, Donna J. Thal, Philip S. Dale, J. Steven Reznick, and Elizabeth Bates. 2007. MacArthur-Bates Communicative Development Inventories: User's Guide and Technical Manual, 2 edition. Brookes.

Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2002. Placing search in context: The concept revisited. ACM Transactions on Information Systems, 20(1):116–131.

Michael C. Frank. 2023. Bridging the data gap between children and large language models. Trends in Cognitive Sciences, 27(11):990–992.

Michael C. Frank, Mika Braginsky, Daniel Yurovsky, and Virginia A. Marchman. 2017. Wordbank: An open repository for developmental vocabulary data. Journal of Child Language, 44(3):677–694.

Daniela Gerz, Ivan Vulić, Felix Hill, Roi Reichart, and Anna Korhonen. 2016. SimVerb-3500: A Large-Scale Evaluation Set of Verb Similarity. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2173–2182.

Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360.

Felix Hill, Roi Reichart, and Anna Korhonen. 2015. SimLex-999: Evaluating Semantic Models With (Genuine) Similarity Estimation. Computational Linguistics, 41(4):665–695.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, and 3 others. 2022. Training Compute-Optimal Large Language Models. In Advances in Neural Information Processing Systems, volume 35.

Jennifer Hu, Alvin Wei Ming Tan, Steven Y. Feng, and Michael C. Frank. 2025. Language production is harder than comprehension for children and language models. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 47.

Michael Y. Hu, Aaron Mueller, Candace Ross, Adina Williams, Tal Linzen, Chengxu Zhuang, Ryan Cotterell, Leshem Choshen, Alex Warstadt, and Ethan Gotlieb Wilcox. 2024. Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. In The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning, pages 1–21.

Philip A. Huebner, Elior Sulem, Cynthia Fisher, and Dan Roth. 2021. BabyBERTa: Learning More Grammar With Small-Scale Child-Directed Language. In Proceedings of the 25th Conference on Computational Natural Language Learning, pages 624–646.

Clifford M. Hurvich and Chih-Ling Tsai. 1989. Regression and time series model selection in small samples. Biometrika, 76(2):297–307.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.

Reinhard Kneser and Hermann Ney. 1995. Improved backing-off for M-gram language modeling. In 1995 International Conference on Acoustics, Speech, and Signal Processing, volume 1, pages 181–184.

Jianhua Lin. 1991. Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1):145–151.

Bria Long, Robert Z. Sparks, Violet Xiang, Stefan Stojanov, Zi Yin, Grace E. Keene, Alvin W. M. Tan, Steven Y. Feng, Chengxu Zhuang, Virginia A. Marchman, Daniel L. K. Yamins, and Michael C. Frank. 2024. The BabyView dataset: High-resolution egocentric videos of infants’ and young children's everyday experiences. arXiv preprint arXiv:2406.10447. Proceedings of Cognitive Computational Neuroscience 2025, DOI: 10.32470/vbbjtb0.

Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, and Daphne Ippolito. 2024. A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3245–3276.

Lillian R. Masek, Sarah J. Paterson, Roberta Michnick Golinkoff, Roger Bakeman, Lauren B. Adamson, Margaret Tresch Owen, Amy Pace, and Kathy Hirsh-Pasek. 2021. Beyond talk: Contributions of quantity and quality of communication to language success across socioeconomic strata. Infancy, 26(1):123– 147.

Robert C. Moore and William Lewis. 2010. Intelligent Selection of Language Model Training Data. In Proceedings of the ACL 2010 Conference Short Papers, pages 220–224.

Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A. Raffel. 2023. Scaling Data-Constrained Language Models. In Advances in Neural Information Processing Systems, volume 36.

Eva Portelance, Yuguang Duan, Michael C. Frank, and Gary Lupyan. 2023. Predicting Age of Acquisition for Children's Early Vocabulary in Five Languages Using Language Model Surprisal. Cognitive Science, 47(9):e13334.

Yulu Qin, Wentao Wang, and Brenden M. Lake. 2024. A systematic investigation of learnability from single child linguistic input. In Proceedings of the 46th Annual Conference of the Cognitive Science Society.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners. Technical report, OpenAI.

Meredith L. Rowe. 2012. A Longitudinal Investigation of the Role of Quantity and Quality of Child-Directed Speech in Vocabulary Development. Child Development, 83(5):1762–1774.

Meredith L. Rowe and Catherine E. Snow. 2020. Analyzing input quality along three dimensions: Interactive, linguistic, and conceptual. Journal of Child Language, 47(1):5–21.

Herbert Rubenstein and John B. Goodenough. 1965. Contextual correlates of synonymy. Communications of the ACM, 8(10):627–633.

Karan Singh, Michael Yu, Varun Gangal, Zhuofu Tao, Sachin Kumar, Emmy Liu, and Steven Y. Feng. 2026. To Memorize or to Retrieve: Scaling the Interaction Between Pretraining and Retrieval. arXiv preprint arXiv:2604.00715.

Wai Keen Vong, Wentao Wang, A. Emin Orhan, and Brenden M. Lake. 2024. Grounded language acquisition through the eyes and ears of a single child. Science, 383(6682):504–511.

Wentao Wang, Wai Keen Vong, Najoung Kim, and Brenden M. Lake. 2023. Finding Structure in One Child's Linguistic Experience. Cognitive Science, 47(6):e13305.

Alex Warstadt and Samuel R. Bowman. 2022. What Artificial Neural Networks Can Tell Us about Human Language Acquisition. In Algebraic Structures in Natural Language, pages 17–60. CRC Press.

Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, and Ryan Cotterell. 2023. Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. In Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, pages 1-34.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. BLiMP: The Benchmark of Linguistic Minimal Pairs for English. Transactions of the Association for Computational Linguistics, 8:377– 392.

Aditya Yedetore, Tal Linzen, Robert Frank, and R. Thomas McCoy. 2023. How poor is the stimulus? Evaluating hierarchical generalization in neural networks trained on child-directed speech. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9370–9393.

Linda Zeng, Steven Y. Feng, and Michael C. Frank. 2026. Bringing Up a Bilingual BabyLM: Investigating Multilingual Language Acquisition Using Small-Scale Models. arXiv preprint arXiv:2603.29552.

## A Training-Recipe Sensitivity

The fixed-recipe analyses use learning rate $3 \times$ $1 0 ^ { - 4 } .$ , dropout 0.1, weight decay 0.1, and 20 epochs. We also evaluated seven alternative configurations varying learning rate $( 1 0 ^ { - 4 } , \ 3 \times 1 0 ^ { - 4 } , \ 1 0 ^ { - 3 } )$ dropout (0.1 or 0.3), weight decay (0.1 or 0.5), and training length (20 or 40 epochs), at 100k–1.6M words for both BabyView and age-2 TinyDialogues and both model sizes. The screening used three corpus draws with seed 0 (480 runs); selected configurations were confirmed with two additional model seeds (120 runs). Selection minimized mean loss across draws on a held-out selection split, without using the reported evaluation sets. The strongestregularization, longer-horizon configuration (learning rate $3 \times 1 0 ^ { - 4 }$ , dropout 0.3, weight decay 0.5, 40 epochs) was selected in 14 of 20 cases; the original fixed configuration was selected in none. Consequently, this bounded grid establishes recipe sensitivity but not a global training optimum.

<table><tr><td>Corpus/model</td><td>Fixed setup Tuned setup</td></tr><tr><td>BabyView micro</td><td>800k-1.6M 400k-800k</td></tr><tr><td>BabyView small</td><td>800k-1.6M 400k-800k</td></tr><tr><td>Age-2 TD micro Age-2 TD small</td><td>400k-800k 200k-400k 200k-400k 200k-400k</td></tr><tr><td></td><td></td></tr></table>

Table 1: Vocabulary-aligned KN-5/GPT-2 crossover ranges at the primary checkpoint. Tuning moves the crossover one tested data size earlier in three of the four comparisons.

The crossover ranges are shown in Table 1 and illustrated for GPT-2 small in Figure 5. A replay of the fixed setup fell within the prespecified 0.005- nat tolerance in 54 of 60 cases (median absolute difference 0.00086 nats; maximum 0.0141), meeting the continuation rule.

## B Power-Law Fit Diagnostics

We fitted $L ( N ) = L _ { \infty } + A N ^ { - \beta }$ to held-out loss at eight sizes (25k–3.2M words), weighting by variance estimates from three corpus draws. Before inspecting results, we specified five criteria for an adequate fit: (A1) parameter range, precision and agreement across starting values; (A2) fit relative to log-linear and log-quadratic alternatives, assessed with the small-sample corrected Akaike information criterion (AICc; Hurvich and Tsai, 1989); (A3) weighted fit and residual patterns; (A4) leave-onedata-size-out predictive accuracy; and (A5) consistent, finite fits across the three draws.

![](images/314102d631d39a5606f7336f7d18d062e6216e14412e54a51a013be81d797b83.jpg)

![](images/f0e0c09057590043a3fd45757df68f0ec42cf7e12ce14bc71d4a9d62e5e21559.jpg)  
Figure 5: GPT-2 small versus KN-5 on the same held-out in-vocabulary units. Positive values mean KN-5 predicts the evaluated units better; negative values mean GPT-2 is better.

The numerical thresholds were $\beta \in [ 0 . 0 5 , 1 . 5 ]$ with interval inside [0.01, 2.0], interval half-width at most 0.5β, and starting-value estimates agreeing within 0.001 (A1); ∆AICc(power − loglinear) ≤ -2 and $\Delta \mathrm { A I C c } ( \mathrm { p o w e r - l o g q u a d r a t i c } ) \le 2 ( \mathrm { A } 2 )$ weighted $R ^ { 2 } \ge 0 . 9 8$ , largest absolute standardized residual at most 2.5, and same-sign residual runs at most 4 (A3); at least 7 of 8 left-out sizes predicted within two standard deviations and mean absolute error at most 1.1 times that of the log-quadratic comparator (A4); and three non-boundary per-draw estimates consistent with the pooled interval (A5).

No core BabyView or age-2 TinyDialogues fit passed the full criteria for either model size at the primary checkpoint. For example, BabyView micro had weighted $R ^ { 2 } = 0 . 9 6 5$ , and AICc was 71.9 for the power law versus 58.5 for the log-linear alternative; representative leave-one-size-out analyses included only 0–2 of eight sizes within two standard deviations. Thus the standard power law did not adequately summarize these observed curves under this stringent test. This does not exclude power-law relationships under other training setups or fitting procedures.

## C KN-5 Vocabulary Alignment and Grammar Controls

KN-5 is a lowercased, utterance-level word model with a closed vocabulary; unseen words map to an unknown-word symbol. GPT-2 instead models their full byte-level spelling and may condition on a longer context. The primary crossover comparison therefore sums both models’ prediction costs over identical held-out units present in the corresponding KN-5 training vocabulary (bits per in-vocabulary unit). This reduces the KN-5 margin, but all four primary-checkpoint crossover ranges are unchanged. At the final checkpoint, BabyView small is the exception: its original closed-vocabulary crossover lies between 1.6M and 3.2M words, while its vocabulary-aligned crossover lies between 800k and 1.6M.

On Zorro, the pooled BabyView-minus-age-2 TinyDialogues contrast was about +1.69 percentage points for KN-5 (95% interval $[ + 0 . 3 4 , + 2 . 9 6 ] )$ versus about +4.51 for the neural models. For negative-polarity-item (NPI) licensing the comparison instead reverses: about —13.9 points for KN-5, versus +32.4 for GPT-2. An item-level lexicalfrequency explanation was not identifiable in the planned test because the intended predictor barely varies across the relevant NPI items; the measurable neural effect is concentrated in the only/even subset. Thus the 5-gram comparison is an informative control, not an architecture-matched ablation or a full explanation of grammatical-benchmark differences. Likewise, KN-5 versus neural ownenvironment advantage magnitudes should not be compared directly under closed-vocabulary scoring. At the seven data sizes where the main Zorro interval excludes zero, per-token scoring agrees in direction at 7/7 sizes for each model; BLiMP agrees at 6/7 for small and 7/7 for micro, with weaker BabyView advantages around 200k-800k words. The full 13-phenomenon breakdown is shown in Figure 6.

## D Cross-Child Specialization Checks

Figure 7 shows the matched neural and countmodel matrices. The overall own-environment advantage was 0.769 BPW (95% interval [0.623, 0.921]; all 16 children positive) for the all-speech comparison. In the non-target-speech evaluation, the own-environment advantage was approximately 0.775 BPW. The lexical decomposition and off-diagonal KN-5 correlation reported in the main text use this non-target-speech evaluation, not the all-speech matrix. Unseen words accounted for approximately 94.8% of the advantage, despite constituting 6.2% of evaluation words; these summaries describe the observed evaluation split and do not separate proper names, household-specific terms, and possible transcription artifacts. The neural and 5-gram matrices correlated at r = .908 after excluding the 16 same-child comparisons (n = 240), and r = .968 across all 256 comparisons, following the same double-centering. The latter magnitude supports agreement in pattern, not agreement in raw loss scale. Further, the off-diagonal relationship in Figure 1B is not driven solely by extreme observations: Spearman correlation is ρ = .883, and removing the two most extreme reciprocal comparisons leaves Pearson r = .864. Recomputing the row- and column-centered matrices after excluding each child in turn yields correlations ranging from .862 to .923.

![](images/3a31a0942d5e96581141f753e0c9b02044c6d6a7d6596267fe926c0ab619fdd2.jpg)  
BabyView – age-2 TinyDialogues, Zorro accuracy (percentage points); pooled 800k-3.2M; two-stage 95% intervals  
Figure 6: Zorro accuracy differences by grammatical phenomenon (BabyView minus age-2 TinyDialogues) for GPT-2 small and micro, pooled across 800k–3.2M training words. Positive values favor BabyView. Error bars are pointwise 95% two-stage bootstrap intervals across three corpus draws and three model seeds per arm; the overall difference is +4.51 percentage points for small and +2.36 for micro.

## E Statistical Procedures and Analysis-Plan Provenance

The replication unit for scaling is one training run (a corpus draw and model seed). Every partialsize corpus draw is a nested trajectory across data sizes: the two-stage bootstrap resamples whole trajectories, then model seeds within the selected draws (2,000 resamples for the main curves). The reported intervals are pointwise 95% intervals conditional on the observed BabyView and TinyDialogues training corpora. Because we have only three corpus draws, our bootstrap uncertainty estimates are limited in precision. We also do not correct for multiple comparisons, so the number of confidence intervals excluding zero should be interpreted cautiously rather than as strong evidence for multiple separate effects. The full-corpus 4.67M BabyView anchor has one fixed corpus draw and three seeds, hence its displayed range reflects seeds only; the size-matched age-2 TinyDialogues anchor has three draws and a two-stage interval. Neither anchor should be confused with the repeatedly drawn 25k-3.2M series.

For individual-environment and persistence analyses, we use child-cluster bootstrap intervals and target-label permutation tests, as specified for each analysis; the persistence interval uses 10,000 childcluster bootstrap resamples. Diversity uses HC1 intervals and Monte-Carlo label permutations over the 63 fixed-budget mixture datasets. The original project analysis plan designated within-child quantity, diversity, between-child heterogeneity, age of acquisition, and child-specific vocabulary as primary. The scaling, age-composition, lexical-locus, persistence, synthetic-heterogeneity, and diversityexplanation analyses were planned later, but their rules were recorded before the corresponding results were examined. These later analyses are not retroactively presented as original primary tests.

![](images/55acc013165b899461554eb84a1e821e6a37be6bed2ef295c89d8dcc2f608782.jpg)

![](images/166fb248df4e6817f6bc59235d6bac4be3967b18ef390a20afe0b5e9eb3bce0a.jpg)  
Figure 7: Cross-child held-out prediction patterns for GPT-2 and KN-5 after the same row/column adjustment. Diagonal entries are masked and colors are clipped to [—0.40, 0.40] BPW to reveal the remaining 240 comparisons, whose patterns correlate at $r = . 9 1$ . The color clipping affects display only, not the correlation.

## F Persistence: Full Analysis and Replication

The primary analysis used 20 children with vocabulary-checklist (CDI) dates defining training cutoffs. Models trained on language collected before each cutoff were evaluated on held-out speech from other people in the same child's environment recorded afterward, using the final checkpoint. There were 67 candidate cutoffs, 65 with comparator sets, and 63 contributing usable future evaluations. The 163 usable future evaluations were grouped by elapsed time (Table 2): 0–3 months (61 evaluations, 20 children), 3–6 months (48, 17), 6– 12 months (35, 10), and more than 12 months (19, 6). Figure 8 shows the original longitudinal results and replication. A pre-cutoff reference window from held-out recordings before the training cutoff averaged 0.853 BPW (65 evaluations across 20 children); it was excluded from the analysis of change over time. The primary analysis required at least 2,000 evaluated words per comparison; a 5,000- word threshold was tested separately. Prediction advantage was defined as the average comparisonmodel loss minus the own-environment model's loss on the same held-out text, in BPW; positive values favor the own-environment model.

Fitting within-child slopes across usable future windows gives a decline of —0.0376 BPW/month (95% interval [-0.0575, -0.0218]), negative for all 16 children with estimable slopes; four children did not have enough windows to fit a slope. An independent chronological-prefix analysis used 17 children, 31 cutoffs, and 91 cells and gave a slope of -0.0321 BPW/month (95% interval [-0.0463, -0.0199]), with 15 of 16 estimable slopes negative. Results were similar with the 5,000-word evaluation threshold (—0.0365), validation-selected checkpoints (—0.0351), and allspeech scoring (—0.0360). Because children also get older as time passes, we cannot tell whether the decline in specialization is due to elapsed time, changes associated with age, or both.

<table><tr><td colspan="2">Months Cells Children BPW [95% CI]</td></tr><tr><td>0-3</td><td>61 20 0.640 [0.574, 0.705]</td></tr><tr><td>3-6 48</td><td>17 0.525 [0.450, 0.600]</td></tr><tr><td>6-12</td><td>35 10 0.456 [0.354, 0.556]</td></tr><tr><td>12+</td><td>19 60.326 [0.206, 0.429]</td></tr></table>

Table 2: Own-environment prediction advantage for later speech from other speakers in the child's environment. Intervals resample children, not individual evaluation windows.

## G Synthetic Differences and Specialization

We constructed controlled synthetic environments using TinyDialogues with the same generation and preprocessing setup. At each mixing level, eight environments each contributed 100k words; three model seeds per environment gave 192 sourceby-target evaluation cases per level (8 training environments $\times \ 8$ held-out environments $\times \ 3$ seeds). The primary manipulation varied a mixture of discourse and participant composition using $\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ . We measured the mean unigram Jensen–Shannon divergence (JSD, bits) between environments rather than assuming that the manipulation produced equal-sized differences.

![](images/cc964621567a47f8201fd23940686cd4920abe6a8ba253a76c32edd7f3d24585.jpg)  
Figure 8: Own-environment prediction advantage over time in the main analysis and an independent chronologicalprefix replication. Points show averages for future evaluation windows and error bars show 95% child-cluster bootstrap intervals. The open diamond is a descriptive pre-cutoff reference, not part of the estimated time trend; exact window counts and slopes appear in Table 2 and the text.

<table><tr><td>Condition</td><td colspan="3">α JSD BPW advantage 95% CI</td></tr><tr><td>Primary</td><td>0.0345</td><td>−.002 [−.030, .028]</td><td rowspan="3"></td></tr><tr><td>Primary</td><td>.25.0426</td><td>.097 [.010, .193]</td></tr><tr><td>Primary</td><td>.50.0650</td><td>.366 [.205, .575]</td></tr><tr><td>Primary</td><td>.75.1012</td><td>.826 [.590, 1.098]</td><td rowspan="2"></td></tr><tr><td>Primary</td><td>1.0.1524</td><td>2.107 [1.690, 2.465]</td></tr><tr><td>Noun/verb</td><td>1.0.0572</td><td></td><td>.289 [.232, .347]</td></tr><tr><td>Noun/category 1.0 .1049</td><td></td><td></td><td>.857 [.494, 1.229]</td></tr><tr><td>Interlocutor</td><td>1.0.0542</td><td></td><td>.976 [.584, 1.633]</td></tr></table>

Table 3: Controlled synthetic-environment specialization. The advantage is other-environment minus own-environment content BPW; larger values indicate stronger specialization. JSD is calculated over the top 5,000 unigram types with add-0.5 smoothing. Intervals resample environments.

Table 3 and Figure 9 show the synthetic manipulation and specialization trend. The advantage grows monotonically with the primary manipulation. At similar measured divergence, the original noun/verb and noun/category controls approximately follow the same pattern as the primary mixture. However, the interlocutor-only manipulation yields about 0.976 BPW advantage at JSD 0.0542, approximately 4.1 times the main-series value interpolated at comparable JSD. Hence, measured unigram divergence is not a universal predictor of specialization across different kinds of distributional change. BabyView's 0.769 BPW advantage at mean JSD 0.106 is compatible with the primary synthetic relationship at comparable divergence, although the effect in BabyView is much more concentrated in unseen vocabulary than in these synthetic manipulations. These results show that differences in the composition of language input can produce specialization, but they do not mean that the synthetic environments capture all the properties of real children's language environments.

## H Raw Scaling for GPT-2 Micro

Figure 10 shows the raw benchmark scaling curves for the 13.9M-parameter GPT-2 micro model, using the same evaluation conventions as the main Figure 4. The full-corpus BabyView anchor has one corpus draw and three seeds (seed-only range), whereas the shared 25k–3.2M series and the sizematched age-2 anchor use three draws and three seeds. The larger age-2-only extensions have no naturalistic counterpart.

## I Word-Similarity and Low-Data Zorro Checks

## I.1 WordSim Evaluation Sensitivity

Our primary WordSim evaluation extracts word representations from the final transformer layer, averages representations across subtokens, and computes Spearman correlation with the human judgments on five datasets. The same extraction rule is used for all trained models. Earlier related work such as Feng et al. (2026) instead used the last subtoken and selected the best layer separately for each test dataset. The current evaluator also uses a different input prefix and a smaller corpusspecific tokenizer. Scores from the two procedures are hence not directly comparable.

![](images/0e409530ab03932a3b4bb0ad524bbd9b430dff8250b48e5a41f6fb11a24ebd0f.jpg)  
Figure 9: Controlled synthetic specialization. (A) Increasing the synthetic mixing parameter α increases the own-environment prediction advantage. (B) The same results plotted against measured excess unigram divergence, alongside three alternative synthetic manipulations and BabyView as a descriptive reference. Excess divergence is the median other-environment minus own-environment JSD. Error bars indicate 95% environment-cluster bootstrap intervals. The BabyView marker is not part of the controlled synthetic experiment. Different types of changes to the language input can produce different levels of specialization, even when the overall differences between environments are similar.

The present evaluations stored correlations at every hidden layer, allowing us to compare layerselection rules without loading or rescoring models. Table 4 compares the mean scores for GPT-2 small trained on all 4.67M BabyView words (one corpus draw, three seeds):

<table><tr><td>Representation/layer convention</td><td>Mean ρ</td></tr><tr><td>Final layer (five tests)</td><td>0.050</td></tr><tr><td>Two-thirds depth (five tests)</td><td>0.082</td></tr><tr><td>MEN-dev-selected (four tests)</td><td>0.044</td></tr><tr><td>Best layer per test (five tests)</td><td>0.128</td></tr><tr><td>Previous work (Feng et al., 2026) (five tests)</td><td>0.153</td></tr></table>

Table 4: WordSim sensitivity to layer selection. The best-layer-per-test-set procedure uses the evaluation targets to choose a layer, so its score is optimistic. The MEN-dev-selected result covers four held-out datasets, unlike the five-dataset means. The v1 result also differs in model, tokenizer, pooling, and input format.

The older best-layer selection rule also increases the score for an untrained BabyView-tokenized small model from -0.007 (final layer) to 0.028. This illustrates why test-set layer selection should not replace the primary fixed-layer evaluation. The archived scores do not separately identify the effect of last-subtoken versus mean-subtoken pooling.

Averaged across training sizes from 800k to 3.2M words, GPT-2 small models trained on BabyView scored slightly lower on WordSim than those trained on age-2 TinyDialogues (a difference of —0.017 in mean Spearman correlation). This difference increased to —0.057 when we restricted the evaluation to word pairs from a shared vocabulary. Similarly, BabyView scored lower than mixed-age TinyDialogues by —0.046 across all word pairs and —0.074 on the shared-vocabulary subset. These two evaluations use different sets of word pairs, so their results should be considered separately. These differences are specific to the evaluated benchmarks and do not establish that BabyView is worse for learning word meanings.

## I.2 Random initialization and 25k-word Zorro

Figure 4A shows the random-initialization references for GPT-2 small, matched to each corpus tokenizer and model seed. These anchors are constant across the displayed training sizes because they refer to the same untrained model/tokenizer configurations; they are not the nominal 50% chance line. They differ between corpora, so training gains should be compared with the corresponding initialization rather than with a single universal baseline. Table 5 summarizes the results at 25k words (three draws and three seeds per corpus).

Open: full/extended sizes  
![](images/c872b7011ac9b4133c8ac41bb31ae227a8cc7d244e021a178de569352a5712af.jpg)

![](images/a9b023f183d857d1b8562a0e3943334ef20110f614323545f47e7cc0b70a1b47.jpg)  
Figure 10: Raw benchmark scaling for GPT-2 micro: (A) Zorro accuracy and (B) mean WordSim Spearman correlation over all 7,917 word pairs in five datasets. Lines connect the shared 25k-3.2M training sizes (pointwise 95% two-stage bootstrap intervals over three corpus draws and three seeds). Hollow markers show the full 4.67Mword BabyView corpus (one draw and a seed-only range), the size-matched age-2 TinyDialogues sample (three draws), and age-2-only extensions at 6.4M and 12.8M words. Color-matched horizontal dashed lines give tokenizermatched random-initialization Zorro baselines (46.1% for BabyView and 54.8% for age-2 TinyDialogues).

<table><tr><td>Corpus / benchmark</td><td>Trained Initial</td><td></td><td>Gain</td></tr><tr><td>BabyView: Zorro (%)</td><td>59.44</td><td>48.41</td><td>+11.03</td></tr><tr><td>Age-2 TD: Zorro (%)</td><td>54.78</td><td>51.98</td><td>+2.80</td></tr><tr><td>BabyView: per-token Zorro (%)</td><td>60.63</td><td>47.71</td><td>+12.91</td></tr><tr><td>Age-2 TD: per-token Zorro (%)</td><td>56.59</td><td>51.22</td><td>+5.37</td></tr><tr><td>BabyView: BLiMP (%)</td><td>54.86</td><td>49.43</td><td>+5.43</td></tr><tr><td>Age-2 TD: BLiMP (%)</td><td>51.35</td><td>51.93</td><td>-0.58</td></tr></table>

Table 5: 25k-word trained versus matched randominitialization performance (GPT-2 small). Gains are in percentage points. The random-initialization reference averages the matched seeds and tokenizer; repeated corpus draws do not create independent initializations.

The BabyView 25k-word improvement is uneven across phenomena. Selected gains over initialization (percentage points) include subjective pronoun case +33.5, negative-polarity-item licensing +29.5, local attractors +26.0, argument structure +21.2, and quantifiers +14.4. In contrast, subject-verb agreement (+0.1), determinernoun agreement (approximately zero), and irregular forms (—0.8) remain near chance; ellipsis declines (—13.7). These are descriptive comparisons, not evidence that broad grammatical rules have been acquired. An existing in-vocabulary Zorro filter gives a similar BabyView result (59.26% versus 59.44% unfiltered; age-2: 54.57% versus 54.78%). For GPT-2 micro, the corresponding random-initialization Zorro references are 46.1% (BabyView) and 54.8% (age-2); these are shown in Figure 10A.

## J Age-Composition Comparisons (Supplementary)

The age-2 TinyDialogues condition consists of child-directed conversations generated for a twoyear-old. The controlled M1 condition combines dialogue targeted at ages 2, 5, and 10 from the same generator family. Its matched comparison with age-2 spans 25k–12.8M words, using three corpus draws and three seeds at each size. M1 continues to 51.2M words without a matched age-2 comparison. Because M2 changes both the target ages and the model generating the conversations, we cannot isolate the effect of age. We include its result at \~190M words for context only.

Figure 11 shows the full age-composition comparison. At 25k and 50k words, mixed-age GPT-2 small had Zorro advantages of about +2.30 and +2.33 percentage points, but at 200k and 400k the differences reversed (-1.50 and

—2.19 points); these apparent low-data deficits weakened in the equal-length item subset. At 3.2M, 6.4M and 12.8M words, the mixed-age small-model advantages were +3.77 (95% interval [1.77, 5.68]), +2.74 ([0.87, 4.52]), and +4.55 ([2.51, 6.38]) points. At larger training sizes, mixed-age data also performed better under the pertoken Zorro evaluation and on BLiMP. The smaller GPT-2 micro model showed a similar pattern, although only near the largest training sizes. Word-Sim combines five benchmark datasets (Rubenstein and Goodenough, 1965; Finkelstein et al., 2002; Hill et al., 2015; Gerz et al., 2016; Bruni et al., 2014). Using the same 7,917 word pairs across five benchmarks for both training conditions, mixed-age TinyDialogues scored higher than age-2- only TinyDialogues on WordSim by 0.061 at 3.2M words, 0.055 at 6.4M, and 0.063 at 12.8M (differences in mean Spearman correlation). Because both conditions use synthetic data, these results do not tell us whether mixed-age input is better or worse than naturalistic child-language input.

![](images/2532aa93b4bdfbde0f9010d8950981acc4ed4278a73f63603b6f824cb9899aa3.jpg)

![](images/3740d7e03662016892c3485a54acc3e01a9ddd10b74288ee92d259a9279d9e1b.jpg)  
Figure 11: Mixed-age minus age-2 TinyDialogues across training sizes: (A) Zorro accuracy differences (percentage points) and (B) word-similarity differences (Spearman). Error bars resample corpus draws and seeds.

In held-out text prediction, mixed-age models increasingly approached the performance of age-2 models on age-2 text as training data increased. However, age-2-only models performed increasingly worse than mixed-age models on held-out mixed-age text. These results suggest that the relative performance of the two synthetic training conditions depends on which language distribution the models are evaluated on.

## K Fixed-Budget Diversity: Supplementary Tests

At the primary 100k-word budget, there were 63 training mixtures and 189 runs. The pooled heldout prediction slope was —0.1236 BPW per unit ln k (95% interval [-0.1356, -0.1116]; Monte-Carlo permutation $p = 0 . 0 0 0 1 )$ , whereas the primary Zorro slope was +0.2003 percentage points per ln k (HC1 95% interval $[ - 0 . 2 8 2 , + 0 . 6 8 2 ] ; p =$ 0.2788). In leave-one-dataset-out analyses, $R ^ { 2 }$ was 0.673 using the number of contributing children (ln k), 0.930 using unigram JSD, 0.719 using vocabulary coverage, and 0.974 using five distributional descriptors together. These results show relationships between training-data properties and prediction performance, but do not establish cause and effect or show whether the findings generalize to new children.

At the secondary 250k-word budget (27 mixture datasets and 81 runs), pooled held-out BPW improved with additional children (slope —0.1465; $p ~ = ~ 0 . 0 0 0 1 )$ The primary Zorro slope was +0.594 points (95% interval [−0.077, 1.265]; p = 0.0854); per-token-normalized Zorro was positive $( p = 0 . 0 4 4 )$ , while BLiMP showed no clear effect (p ≈ 0.37). These supplementary tests expand the Section 5 analysis. Hence, the primary 100k-word conclusion should not be generalized to a claim that increasing diversity never helps grammaticality.

## L Child-Speech Removal Control

As a control for Section 4.2, we compared two training conditions for 10 children with enough recorded speech: removing the target child's own utterances, or removing the same amount of speech at random. Each condition used three model seeds, and results were compared within children and seeds. On pooled held-out BabyView text, removing the child's speech resulted in 9.761 BPW, compared with 9.504 BPW under random removal (difference +0.257 BPW; 10/10 children; exact signflip $p = 0 . 0 0 2 )$ . On held-out text from the child's own environment, the corresponding values were 8.845 and 8.562 BPW (difference +0.283 BPW; $p = 0 . 0 0 2 )$ . In contrast, Zorro accuracy was similar between conditions (57.89% versus 58.02%; difference —0.12 percentage points, $p = 0 . 7 7 )$ , as was BLiMP (52.02% versus 52.25%; difference —0.24 points, $p = 0 . 2 8 )$ . Thus, the child's own speech contributes useful information for predicting heldout language, but we find no corresponding improvement on these grammatical benchmarks.

## M Developmental Outcomes (Supplementary)

Following earlier work on predicting when children learn words (Chang and Bergen, 2022; Portelance et al., 2023), we tested whether the order in which language models learn words helps predict children's age of word acquisition in Wordbank (Frank et al., 2017). Our baseline already included word frequency, concreteness, and word class. Adding the model's learning order did not improve predictions according to our predefined criterion $( \Delta \mathrm { A I C } ~ = ~ + 0 . 6 6$ , reported interval $[ - 2 . 9 1 , + 1 . 5 1 ]$ ; improvement required a value below -2). The additional predictor also had only a small correlation with children's age of acquisition after accounting for the baseline factors $( r = - 0 . 0 8 5 , 9 5 \%$ interval [-0.154, -0.015]).

In a separate analysis using child-specific vocabulary reports (CDI; Fenson et al., 2007) from 20 children (65 reports), models trained on a child's own language environment showed a small advantage in distinguishing words the child could produce from words they could not $( \Delta \mathrm { A U C } = 0 . 0 2 2 9$ permutation $p = 0 . 0 0 0 6 )$ . However, a predictor based only on word frequency performed slightly better $( \Delta \mathrm { A U C } = 0 . 0 2 9 8 )$ , and the model's advantage disappeared and reversed after accounting for frequency $( \Delta \mathrm { A U C } = - 0 . 0 1 4 7 )$ . These results do not provide convincing evidence that our models capture how children actually learn vocabulary.

## N Computational Resources and Implementation Details

Hardware and compute. Model training and most GPU evaluations ran on eight single-node RunPod instances with 1-8 NVIDIA H100 80 GB or 2 NVIDIA H200 GPUs (CUDA 12.8). A local workstation with an RTX 3090 Ti 24 GB handled two additional GPU evaluation jobs. Corpus preparation, tokenizer training, KN-5 baselines, and statistical analyses used workstation or pod CPUs. The experiment ledger accounts for approximately 429.5 GPU-hours of training and GPU-based evaluation: 411.8 hours in experiment families contributing to this paper and 17.7 hours in pilots or other investigations (Table 6). GPU time was estimated from 10-second nvidia-smi activity samples where available; for some jobs we instead sum process wall-clock times, which can overcount overlapping GPU work. Thus, these are approximate, not uniformly measured, GPU-active hours. The eight GPU pods were allocated for approximately 807 GPU-hours from creation to termination (one creation time is estimated), including setup, CPU-only work, and idle periods. CPU-only compute was not separately metered.

<table><tr><td>Experiment family</td><td>Runs</td><td>GPU-h</td></tr><tr><td>Main scaling and closeout checks</td><td>390</td><td>35.4</td></tr><tr><td>Mixed-age TinyDialogues scaling</td><td>636</td><td>239.6</td></tr><tr><td>Large-data replication (plus 24 re- plays)</td><td>24</td><td>14.0</td></tr><tr><td>Recipe sensitivity grid</td><td>600</td><td>20.4</td></tr><tr><td>Individual environments and prefixes</td><td>525</td><td>20.9</td></tr><tr><td>Diversity, reference data, and AoA</td><td>494</td><td>23.8</td></tr><tr><td>Robustness experiments</td><td>534</td><td>18.3</td></tr><tr><td>Pre-CDI child models</td><td>201</td><td>20.9</td></tr><tr><td>Controlled synthetic specialization</td><td>244</td><td>11.0</td></tr><tr><td>GPU re-scoring and persistence (no new models)</td><td></td><td>7.3</td></tr><tr><td>Pilots and other investigations</td><td>≈1,500</td><td>17.7</td></tr><tr><td>Total</td><td>≈5,100</td><td>≈429.5</td></tr></table>

Table 6: Approximate GPU-work accounting. GPUhours combine sampled utilization and process-time proxies; training-run counts exclude the separate replay jobs. Evaluation time is included. Allocated RunPod time was approximately 807 GPU-hours.

Models and optimization. We use Python 3.11, PyTorch 2.11.0+cu128, transformers 5.12.1, tokenizers 0.22.2, and NumPy 2.4.6. GPT-2 micro has 6 layers, 8 attention heads, and width 384; GPT-2 small has 12 layers, 12 heads, and width 768. All use 1,024-token contexts and tied embeddings; parameter counts vary with vocabulary size as reported in §3.2. AdamW uses $( \beta _ { 1 } , \beta _ { 2 } ) =$ $( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ , and weight decay on matrixvalued parameters only. Each optimizer step uses 16,384 tokens (16 blocks of 1,024), without gradient accumulation, with gradient clipping at norm 1. The learning rate warms up linearly over 3% of steps and then decays by a cosine schedule to 10% of its peak. Training uses bf16 autocast and fp32 master weights. The prepared block order is fixed across passes, without reshuffling; model seeds are 0–2. The held-out selection split is scored every 5% of training steps. Fixed-recipe and tuned hyperparameters are reported in Appendix A.

Tokenizers and baseline. Tokenizers use bytelevel BPE (GPT-2 pre-tokenization; minimum frequency 2). Speaker tags and utterance separators are added as special tokens. The shared 16ktype tokenizer is used for individual-child analyses; the 7.5k-type scaling tokenizers are trained on corpus-specific reserved splits excluded from model training and final evaluation. KN-5 uses our own modified interpolated Kneser-Ney implementation (Kneser and Ney, 1995; Chen and Goodman, 1999), with three count-based discounts per order and a uniform base reserving mass for <unk>. Text is lowercased and whitespace-tokenized; n-gram contexts do not cross utterance boundaries.

Evaluation. Content bits per word are obtained by summing negative log-probabilities over content tokens in non-overlapping 1,024-token windows, dividing by ln 2 and by the word count; format tokens are excluded. Zorro and BLiMP minimal pairs are scored by summed full-sentence log-probability, with ties counting as half credit; BLiMP uses the BabyLM-2026 vocabulary-filtered release. Word similarity uses final-layer hidden states averaged across subtokens, cosine similarity, and Spearman correlation. Appendix E gives resampling methods; analysis scripts record seeds and repository commits. We use transcripts from the BabyView 2026.1 release with our own quality, language, and speaker filters, without rerunning the upstream transcription pipeline. TinyDialogues data are reused from prior work (Feng et al., 2024).