# The Value of Not Knowing: How Fresh Eyes through Naive LLMs Fuel Creativity

David B. D’Ambrosio, Alon Albalak, Joel Lehman, Kenneth O. Stanley

## Abstract

Many major scientific advances come from approaching a problem with a new or novel perspective. While large language models (LLMs) with internet-scale knowledge seem poised to take such perspectives, their consistent generation remains elusive. Instead, models are drawn to dominant views and preexisting, infield bias. To explore the potential of fresh eyes to increase creativity, this paper proposes naive LLMs, models that are made intentionally ignorant of a specific topic. Through deprivation in training or intentional forgetting, naive models are unburdened by existing dogma, allowing them to be free to make new associations and conjectures. These naive models can then be paired with a stronger mentor model that can refine their unique perspectives into concrete proposals. For the 1B and 7B models tested in this paper, naive models consistently produce seed ideas that are more novel and useful than baseline counterparts, and mentor refinement improves the ideas even further. Thus, contrary to the prevailing tradition of scaling up training data, we demonstrate a case where less information is actually more powerful, in particular when the aim is to discover genuinely outside-thebox solutions.

If we approach a task with ignorance, it can remove the barricade of knowledge blocking progress.

—Rick Rubin, The Creative Act (Rubin, 2023)

## Introduction

With so much emphasis today on scaling laws (Kaplan et al., 2020; Subramanyam et al., 2026) and data acquisition (Villalobos et al., 2024), the possibility that less data or knowing less could ever be advantageous is rarely discussed. Yet there is at least one intriguing precedent with humans for questioning the assumption that more is always better: in particular, students or newcomers to fields often benefit from the novel perspective possible through fresh eyes. Indeed, throughout the history of science there are many famous examples of newcomers to a field who made breakthrough discoveries because they were unencumbered by typical biases or brought a fresh perspective from a different area of study (Jeppesen & Lakhani, 2010; Azoulay et al., 2019; Kuhn, 2012).

The advantage of fresh eyes is particularly provocative in the context of today’s relentless quest for ever more data and scale. What does it mean for LLMs that it is sometimes helpful for humans to be naive in a particular area, particularly in the pursuit of creative insight (Wiley, 1998; Jeppesen & Lakhani, 2010)? While the ingredients for unconventional creative insight may in principle exist within the weights of a large model, overwhelming familiarity with the dominant paradigm may make such connections prohibitively unlikely.

The issue is not that current models are incapable of all creative insight. LLMs have produced research contributions across multiple fields (Alon et al., 2026; Tsoukalas et al., 2026; Guevara et al., 2026; OpenAI, 2026b). Rather, the opportunity is that some kinds of insights are more likely through fresh eyes, and those are the types that current monolithic models are not suited to discovering. The possibility of such “naive insight” represents a particularly delicate research opportunity because it pushes against the tidal wave of scaling and data that currently propels almost every measure of progress in the field. After all, the bitter lesson (Sutton, 2019; Kaplan et al., 2020; Hoffmann et al., 2022) is often interpreted to imply that more computation and more data will continue to benefit models. To gain even a foothold on this opportunity to explore the implications of fresh eyes requires at least a partial abandonment of well-understood best practices and the idea that “knowing everything” is always the best approach.

So in what way should the potential of fresh eyes be explored? Prompting a model to approach problems pretending it has no knowledge is an option, but models have difficulty suppressing what they know when instructed to (Li et al., 2026b), and even when made to underperform on command they only imperfectly imitate a genuinely less capable model (van der Weij et al., 2025); the information is still there and will ultimately influence the response. Results in this paper further confirm the limitations of such a simplistic prompt-based approach.

As an alternative, the approach that follows draws inspiration from the notion of an outsider or young person entering a new field by introducing the concept of a naive LLM, which means an LLM that is unaware of a specific topic while maintaining a scale-appropriate understanding of everything else. The idea that the model should still be competent outside the area of ignorance is important: we would not expect innovative thinking to emerge from generally diminished cognition. On the contrary, the idea is that an otherwise competent intellect is particularly empowered when unleashed into a new area (to it) with fresh eyes, unlike standard models that have been shown to be drawn to existing distributions (Zhang et al., 2026; Kirk et al., 2024).

In this way, naive LLMs differ from other attempts at limited models such as LittleLearner (Li et al., 2026a) and the GPT-1900 model (Hla, 2026) in the sense that naive models still represent a capable individual with modern knowledge and can apply these vast capabilities to an “unfamiliar” domain. By limiting the model’s knowledge selectively rather than broadly, it can draw on a breadth of skills and expertise that can be applied free of preconceptions to its domain of ignorance.

Furthermore, we also explore the potential for the advantage of fresh eyes to be refined with the help of a more knowledgeable mentor. While a naive entrant to a field may have interesting seed ideas, refining those ideas into more practical proposals can take some guidance. The intriguing possibility is that once that guidance is provided, the advantage of coming from a fresh perspective may be made more practical.

Two different methods for eliciting naive behavior from LLMs are explored: training from scratch and forgetting. These approaches are tested across model sizes, model families, and topics. The compute necessary to train a new model from scratch or even to make it forget a topic means that naive networks here are restricted to smaller 1B and 7B sizes, but the results still provide a strong hint that less can be more. As long as topics are not beyond the cognitive capacity of these small models, a consistent trend emerges suggesting the very unusual result that more ignorant models are genuinely scoring higher in creative measures and separating themselves farther out in embedding space. Mentors also help refine the utility and feasibility of the naively originated ideas.

The aim of this work is not to be prescriptive, that is, there are many possible definitions of and ways to make a naive LLM, even beyond what is introduced here. Instead the results that follow shine a light on the question of whether it is better to know “everything,” and suggest that there are potentially ways to benefit from having less knowledge, especially in the domain of creativity. While only a start, these results matter because they raise the intriguing question of what a frontier model that had never seen the oncology literature might propose about cancer research.

## 2 Background

This section reviews topics related to success with limited knowledge, including a general scientific view, past work in machine learning on systems with limited knowledge, and attempts to reduce the knowledge of existing models.

## 2.1 The Downsides (and Upsides) of Knowledge in Science

The Wright Brothers ran a bicycle repair shop and lacked any formal engineering training, and yet they managed to create the world’s first true airplane (Jakab, 1990). Science is filled with such examples of approaching problems with “fresh eyes” and proposing new solutions unbounded by the typical biases and beliefs of the field. However, these are not just black swan anecdotes; research has shown that it is exactly these new perspectives that lead to breakthroughs. Wiley (1998) demonstrates that strong existing domain knowledge can constrain experts to “an area of the search space in which the solution does not reside” when confronted with problems related to that domain. Additionally, an analysis of a scientific bounty program, InnoCentive, showed an unintuitively positive correlation between a solver’s success and the distance between the field of the question and the field of the scientist answering it, on problems that had resisted traditional approaches (Jeppesen & Lakhani, 2010). Furthermore, when established scientists pass away there is a documented surge in new and novel approaches in their field (Azoulay et al., 2019).

These new and novel approaches typically come from outside the establishment; either from other fields or from young scientists entering the field. Indeed, inventive and scientific disruptiveness declines with age (Kaltenberg et al., 2023; Cui et al., 2026). Younger researchers are also more likely to depart from existing approaches (Packalen & Bhattacharya, 2019). Still, peak research output remains later in life and its age has been trending higher in recent decades (Jones, 2010), which is attributed to (human) training time. So competency and creativity move in opposite directions, while the growing corpus of knowledge means sacrificing a researcher’s most creative and open years to additional study. This paper decouples these seemingly opposing forces into the naive and mentor roles.

Uzzi et al. (2013) studied millions of papers in an attempt to determine what makes research successful, and found that the highest-impact work pairs a conventional base with a few unusual, novel combinations. In fact, exposure to a persistent minority viewpoint can stimulate divergent thinking in a majority and lead them to correct solutions the minority never even offered (Nemeth, 1986; Nemeth & Wachtler, 1983), and diverse groups can sometimes outperform skilled ones (Hong & Page, 2004). These examples all show that ideation unencumbered with conventional dogma can sometimes unlock productive new research directions.

## 2.2 Machine Learning with Limited Knowledge

The advantages of naivety and outsider perspectives sometimes appear in machine learning as well. Perhaps the most famous example is AlphaGo Zero (Silver et al., 2017). In contrast to standard AlphaGo, it learned to play Go only from self-play, without exposure to human games. This limitation resulted in unique and previously undiscovered strategies because it was not influenced by centuries of human play.

More generally, more data is not always better; forcing models to operate on limited memory can improve performance in language learning (Thamma & Heilbron, 2026). Penedo et al. (2024) demonstrated that a model trained on FineWeb-Edu, a well-curated 1.3T-token educational subset of the 15T-token FineWeb dataset, outperformed models trained on the full FineWeb dataset on common benchmarks like MMLU (Hendrycks et al., 2021) and ARC (Clark et al., 2018). The difference between these examples and the naive LLM introduced in this paper is that these past works mainly target efficiency rather than creativity.

Recently there has also been interest in training language models with a limited corpus. Capping knowledge at a 5th grade level (Li et al., 2026a) or the year 1900 (Hla, 2026) produces models that have interesting perspectives and unique applications, e.g. to test curriculum or teaching methods, or to probe whether later discoveries can be reached from earlier knowledge. However, such models are universally limited, whereas the models in this paper are selectively limited. In this paper, limiting only a portion of the model’s capabilities allows the reweighting of existing information away from standard biases (Zhang et al., 2026) instead of a blanket limit. These reweightings align with approaches like novelty search (Lehman & Stanley, 2011) and quality diversity (Pugh et al., 2016), where the goal is to find different solutions rather than just the best one.

## 2.3 Machine Unlearning

The main approach to achieving selectively naive LLMs in this paper is through removing data from training either so the model has never heard of the topic to begin with or to cause it to forget something it has already learned. The former has been shown to be effective but prohibitive (O’Brien et al., 2026) and the latter echoes prior “unlearning” approaches in the field of Machine Unlearning whose motivation is that retraining from scratch each time data needs to be scrubbed from the model would be prohibitively expensive. The ultimate goal in these past approaches is usually to remove data for different purposes: legal, such as privacy (Jang et al., 2023) and copyright (Eldan & Russinovich, 2023); safety, such as for bio-weapons (Li et al., 2024) and false information (Lu & Koehn, 2025); or alignment, like bias (Dige et al., 2024) and explicit content (Gandikota et al., 2023). Thus while there are many potentially relevant approaches that focus on removing training examples (Liu et al., 2025), the methods in this paper do not require such rigorous guarantees; no one will be sued if the naive model talks about an idea that turns out not to be novel.

Other approaches steer output at inference time by adding activation vectors computed from contrastive prompts (Turner et al., 2024; Rimsky et al., 2024). Standard unlearning methods all proved challenging in preliminary experimentation to directly apply to our domains. This limitation is directly called out by the authors of one of the most influential unlearning examples, which was able to “unlearn” the concept of Harry Potter (Eldan & Russinovich, 2023). They note that it remains uncertain whether their technique extends to non-fiction content, where there are fewer idiosyncratic terms and concepts are more embedded in broader ideas. Real-world entities are likely to be even harder to forget; for example, forgetting Albert Einstein would touch not only on the obvious contributions to science and math, but also sociopolitical artifacts like lists of Nobel Prize winners, the Manhattan Project, and pop culture references and slang.

## 3 Method

This section explains both the creation of naive models and the pipeline for generating robust ideas that combines the output of the naive models with the refinement of the mentor.

## 3.1 Naive Generation

Potential approaches to generating naive output from a model vary in cost, effectiveness, and practicality. This paper explores trainingfrom scratch (true naive) andforgetting (both in Figure 1).

Training from Scratch: True Naivety. The purest form of naivety is to have genuinely never encountered a particular topic at all. The only way for that to happen is to train from scratch with the topic removed from the dataset. This approach is computationally expensive and becomes prohibitive as model size increases. It simply would not be feasible to train a new frontier-level model from scratch for every area of study. However, it is still important to explore this approach as an upper bound for what a naive perspective can achieve: what would interacting with such a model be like and how would its output be different? For practical purposes only 1B scale models (specifically a 26-layer nanochat model from Karpathy, 2025) are tested by training from scratch, but the process is general enough that it could be applied at any model size if compute were available. The main approach is to run a filter over every piece of data before the model sees it.

Removing the topic from the dataset is achieved through a combination of basic word matching and conditional regular expressions. For a given topic, a human and an LLM (claude-opus-4-5-20251101, Anthropic, 2025) collaboratively design the filter’s terms and criteria. The filter is blunt: if a data point is matched, the training example is discarded. While simple in principle, such filtering is nontrivial because words and acronyms have multiple meanings, small words can fit within other words, and proper nouns like brand names can be ambiguous. Additionally, the boundaries of a topic can be fuzzy, e.g. fluoride is most associated with dental care, but it also has many other uses that we would not want to remove when making a model naive to dental hygiene, especially if we want to leave open the possibility of it rediscovering the use of fluoride from scratch. So while the filters in this paper pass through multiple rounds of refinement, they are not and probably cannot be perfect. They are, however, still sufficient to show significant impact with limited effect on other topics.

![](images/1ebfbd3b70f7132c40a37bc96d955d0a7c3a8d6bdfb317c76fae1d62d32bcc1a.jpg)  
Figure 1 Methods for obtaining naive LLMs. There are a variety of methods to make a model unfamiliar with a topic. This paper explores training from scratch and forgetting (top two rows), and compares against baselines prompted for naivety and queried directly. All approaches proceed the same way: (1) Train (or further train) a model. (2) Generate a “seed” idea with the model. (3) Give the seed idea to a mentor that refines the seed into a concrete proposal. The main difference among the approaches is the point at which the topic is withheld from the model.

Forgetting. The cost of retraining a model from scratch, even at modest sizes, is prohibitive. However, as mentioned in Section 2.3, there is a field dedicated to removing existing knowledge from already trained networks called machine unlearning (Liu et al., 2025). Such kinds of intervention usually happen after the costly pre-training step and thus may be a good middle-ground. During preliminary research, standard unlearning techniques (Liu et al., 2025; Eldan & Russinovich, 2023) and related approaches like activation steering (Turner et al., 2024; Rimsky et al., 2024) tended towards incoherence or refusal behaviors on chosen topics (which is understandable given the divergent end goals of this paper and general unlearning), so the main study in this paper does not explore them further, but more advanced or nuanced approaches may be applicable with more research.

As a practical alternative we found an effective yet unconventional approach to intentional forgetting based on a phenomenon LLM training typically tries to avoid: catastrophic forgetting (McCloskey & Cohen, 1989). The key phenomenon is when a model loses knowledge or a skill during extended training (typically post-training) once it is no longer exposed to the original knowledge (Luo et al., 2025). Because the goal of training naive LLMs is different from typical unlearning (e.g. there are no legal or ethical repercussions to retaining the data), this approach offers the ability to gradually erode information and connections in a way that loses fine details but may retain surface-level information. This kind of partial forgetting is reminiscent of a real naive person who may have casually heard of a topic, but does not understand it in depth.

Practically, forgetting is implemented with a model that has gone through a standard pre-training process by applying the filter described above to only the SFT dataset. To achieve the forgetting, more SFT epochs than normal are applied and thus training continues until performance on a topicspecific dataset falls below a threshold, typically two to three epochs beyond the standard training recommendations for a 7B model. However, anecdotally, even after ten or more additional epochs of training the model remained coherent, unlike the other unlearning approaches attempted.

Prompting. As a control, we can also simply ask the models to be naive. Asking a model to adopt a persona or take a specific viewpoint is a common prompt-engineering technique (Lutz et al., 2025). While this approach offers a low-cost solution, a challenge is that LLMs cannot reliably suppress prior knowledge on instruction (Li et al., 2026b) and only imperfectly emulate weaker models (van der Weij et al., 2025); simulating true naivety is difficult when you are already an expert. To implement this approach, a baseline version of the model that was trained without any filtering has its prompts prepended with, “Pretend you have never heard of {\$topic\_description} and answer the following question as such:”. The other control in this paper is to simply ask the baseline model the question directly, creating a baseline response to compare naive approaches against.

## 3.2 Mentors

While the naive approaches can generate interesting seed ideas, almost by definition, their lack of knowledge about the topic means their ideas may need refinement, which raises the idea of a mentor model. By giving a strong mentor model the naive model’s seed idea, the mentor can be inspired to fill in holes or inconsistencies in the naive model’s initial thoughts. In this paper, Gemini 3.1 Pro (Google DeepMind, 2026) refines the seeds, although any sufficiently capable model could work.

The process for mentoring is straightforward (Figure 1): a model generates a seed response to the initial question and then that seed is passed on to the mentor model with framing text (Appendix A). The mentor is specifically instructed to follow the seed and it is later judged on this alignment; otherwise, strong models were found to discard an unconventional seed and invent their own, “bet ter” approach whole cloth. The mentor is also instructed to develop only the most interesting idea if a seed response contains multiple possible directions. Otherwise the mentors would again gravitate towards more standard, in-distribution options if they have to pick among multiple options. Unchecked, the mentors always preferred to steer the discussion to more conventional approaches, but when they are forced to engage with the naive seeds they can refine them into unique final ideas.

## 4 Experiments

In this paper, naive models are generated across a variety of topics (one model per topic, per size) and asked to solve problems relevant to that same topic. This section details the models trained and tested, how topics are generated, and how the responses are evaluated.

## 4.1 Models

Naive models are explored at 1B and 7B sizes. Models follow their recommended training recipes (see Appendix B for details) with the only change being the data filtering described in Section 3.

The smallest model is a depth 26 nanochat (Karpathy, 2025) (roughly 1B parameters) that is run with pre, mid, and SFT training; the optional RL step is not applied. The nanochat model was chosen for its open framework and fast iteration speed. For 7B experiments, the process starts with allenai/Olmo-3-1025-7B, which is already pretrained on an unfiltered dataset and is then further trained with its standard Instruct SFT dataset (Dolci-Instruct-SFT) (Team Olmo, 2025) and no further training stages. The Olmo family was (again) chosen for the open framework and training process. Gemini 3.1 Pro serves as the mentor. The fact that Gemini is a different model family from the others benefits it in the role of a mentor because in some cases LLMs are biased to their own output (Panickssery et al., 2024). That said, any sufficiently capable frontier model could take on the role.

## 4.2 Naive Topics

Because of the manual curation of training data filters and questions, this study is limited to four topics. It may be possible to automate the filtration process in the future, but initial tests suggest that topic fuzziness and scope benefit from human oversight to be effective. The four topics are dental hygiene, neuroevolution, vaccination, and backpropagation. They were chosen to cover a breadth of complexity levels and levels of required technical knowledge. For example, dental hygiene is common knowledge that most people would be expected to have some familiarity with, whereas neuroevolution (Stanley et al., 2019) is a somewhat niche machine learning field (that the authors are familiar with and qualified to make nuanced judgments on).

However, importantly, all of these topics have open problems and challenges where fresh eyes can potentially help. For each topic, in addition to the filter terms, a set of topic-specific problem-solving questions is generated (Appendix C). These are roughly of the form, “I am experiencing {problem in the field}. Can you help me solve it?” or “I need help with {thing the field helps solve}. Can you help me invent something to solve it?” The specific formulation varies by the topic, but the goal is to elicit a helpful seed response from the model that can be practically refined by the mentor. For that reason, simple definitional question like, “What is a toothbrush?” are avoided; the responses to such questions can be novel or even entertaining, but there is nothing for the mentor to refine.

## 4.3 Evaluation

Seed ideas and mentor refinements are evaluated in several ways: an LLM judge produces qualitative metrics, embedding distance further assesses distributional shifts quantitatively, and distributions are also analyzed qualitatively through lexical features. Multiple metrics are helpful because no single measure is reliable on its own: LLM judges can be biased, embedding distances can conflate content with style, and lexical features show what differs but not whether it matters.

Idea novelty is a key inspiration for exploring naive output; in real life, approaching a problem with “fresh eyes” is a great way to break out of typical patterns. Yet a subjective metric like “novelty” is notoriously difficult to measure precisely (Lamb et al., 2018). LLM judges serve as an emerging option (Zheng et al., 2023) but still have issues that must be mitigated. In this paper, judges rank all responses (seeds and refinements) on three criteria on a scale of 1–10 (Appendix E includes rubrics): novelty, utility, and feasibility. The latter two are meant as a sanity check on the first; it is easy to come up with a novel but useless or impractical idea. It may even be expected that the naive models do so because they lack technical grounding. The job of the mentor is to ensure the proposal is sound across all the metrics, although the model is not explicitly told about the metrics. Mentor refinements are additionally assessed on faithfulness to the seed prompt to avoid the mentor substituting a different idea if it does not like the seed. To balance cost and effectiveness (cheaper but less capable judges struggled with the nuanced judgments required), GPT-5.6 Luna (OpenAI, 2026a) is always the judge. An additional benefit is that Luna belongs to a different model family than all the other models in the experiments and does not suffer from in-family bias mentioned earlier.

Embedding distances are a quantitative measure of how much a naive model’s response deviates from a non-naive baseline response of the same model type. All responses are encoded with Qwen3-Embedding-0.6B (Zhang et al., 2025). Because the prompts are all related to the same topic, embeddings are residualized before comparing to remove the encoded topic description by subtracting the topic direction (cf. iterative nullspace projection (Ravfogel et al., 2020)) and then unit-normalized. The seed-paired average cosine distance,

$$
\bar { d } _ { \mathrm { c o s } } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \left( 1 - \frac { \mathbf { u } _ { s } \cdot \mathbf { v } _ { s } } { \| \mathbf { u } _ { s } \| \| \mathbf { v } _ { s } \| } \right) ,\tag{1}
$$

between treatment responses (trained naive, prompted naive, and baseline) is computed, where S is the number of random seeds, and $\mathbf { u } _ { s }$ and ${ \bf v } _ { s }$ are the embeddings produced under seed s by the two conditions being compared. They are compared with PERMANOVA with 9,999 label permutations to compute group differences (Anderson, 2001). To determine which method is further from the baseline, a one-sided relative MMD test (Bounliphone et al., 2016) is applied. Because UMAP and t-SNE 2D projections can distort distances and cluster structure (Wattenberg et al., 2016; Chari & Pachter, 2023), the embeddings are instead visualized by projecting onto a single linear direction via group means, following the difference-of-means projections used to assess class separability in LLM activations (Braun et al., 2025). Comparing the seed ideas to their mentor refinements would also be interesting in principle, but the refinement is generated by a different model family and size, which typically dominate embedding differences and so these comparisons are not performed.

Because embeddings can conflate things like content and style (Wegmann et al., 2022) and singlevector embeddings have fundamental limits in representing fine-grained distinctions that standard benchmarks do not reveal (Weller et al., 2026), lexical features are also explored. Specifically through the “Fightin’ words” approach (Monroe et al., 2008) that looks at the distinctive terms from each group and ranks them by the z-scored weighted log-odds ratio with an informative Dirichlet prior. Positive values are over-represented in the named group and negative values are underrepresented with shared vocabulary shrunk toward zero. This analysis is applied to the mentor refinements to compare the final ideas.

## 5 Results

Results are shared first for truly naive 1B models that are trained from scratch, followed by an examination of forgetting in larger 7B models. For each model size and topic explored, a new naive LLM is trained and 32 seed ideas are generated for each of three topic-dependent prompts from the naive, baseline, and prompted naive (i.e. the baseline model with instructions to pretend it does not know the topic) models. Then these seeds are refined by the mentor (frontier model) and both the seed and refinement are evaluated.

## 5.1 Truly Naive 1B Models

At the 1B scale, it is possible to test truly naive models by excluding specific topics from the training set. Results are compared to a baseline model that was trained without any topic filtering and to the same baseline simply prompted to be naive. In the dental hygiene domain, the from-scratch naive model seeds are scored as significantly more novel (Figure 2a) than the baseline and prompted models $( p ~ < ~ 0 . 0 0 1$ for both), although they are also rated significantly less feasible than both $( p \textless 0 . 0 0 1 )$ . When the mentor is applied, the difference in feasibility is removed (i.e. the naive proposals are made as scientifically plausible as baseline, standard proposals) and the from-scratch naive approaches become even more novel, retaining their significant lead $( p < 0 . 0 0 1 )$ ). Accordingly, the from-scratch naive model produces significantly different outputs than both the baseline and prompted naive models $( p < 0 . 0 0 1$ for both seeds and refinements, Figure 2b) overall, although the distributions specifically for prompt 2 overlap heavily. The implication is that training from scratch can indeed affect model output on a specific topic. Also, the mentor is able to enhance the novelty while finding practical use cases that the naive model is unaware of.

![](images/835416b8582a17e39c8c880fe80dd955a2253c5cf865835e4cf1d3457ca9ad3e.jpg)  
(a) Dental Hygiene Judge Scores

![](images/f97342c8821b8e85477e728210e4cd9396b4e7140779f4f7c546ff28e51abbaf.jpg)  
(b) Dental Hygiene Response Embeddings

Figure 2 1B naive models outperform baselines on dental hygiene ideas. The judge scores are shown in (a), averaged across all prompts for a topic. All seed scores are shown in solid colors and mentor refinement scores are hashed. Bars show mean score across all prompts; error bars show 95% CI. For the relatively simple topic of dental hygiene, true naive models produce more novel, but less feasible seed ideas (a). After mentor refinement, feasibility equalizes and novelty further improves. The embedding images (b) reflect the PER-MANOVA and MMD analyses. They are represented as 1-D projections of individual outputs (bottom ticks) through the three group means (dashed lines) following from difference-of-means projections in analyzing LLM class separation (Braun et al., 2025). Curves are per-group densities with less overlap indicating greater separation. They demonstrate significant distances between the naive (blue) and both baseline approaches for both initial seeds and refinements, although in some cases (e.g. as in prompt 2) the baseline and naive overlap substantially.

However, the mentor’s refinement is not without a cost; while the differences in embedding space remain significant in this model, the mentor can reduce them. One reason for this reduction is that the mentor model actually knows about the topic at hand and is thus free to incorporate common terms and ideas, e.g. “standard dental practice relies on nylon bristles and waxy floss...” Additionally, the scientific grounding can remove some of the more fantastical or speculative elements, e.g. vague appeals to “AI-powered” sensing and “advanced nanotechnology” in 1B naive seeds are typically dropped while the underlying physical concept (e.g. an in-mouth device or micro-saws) is retained and given a concrete mechanism. Furthermore, the refined responses are all produced by the same, untreated frontier model, which likely reduces any stylistic or syntactic deltas. Ultimately, the models still produce different ideas, as shown by the judge scores for the mentor refinements (Figure 2a) and by examining the lexical features of the output (Figure 3). For example, the prompted and baseline both heavily reference common dental implements (e.g. floss and toothbrushes) while the naive models focus more on techniques and methods.

Qualitatively, the baseline model tends to propose extensions to existing dental hygiene products like a floss holder or to add features like Bluetooth to a toothbrush. The prompted naive model either gives definitional answers (e.g. “a toothbrush is...”) or sometimes even ignores the “pretend you don’t know” instructions entirely. In contrast, the from-scratch naive model proposes more radical inventions like a mouth-cleaning drone, or a “tongue stone” that cleans teeth hands-free by being pushed around with your tongue. When given to the mentor the novel ideas are further elaborated and made more practical: in the case of the “tongue stone” the mentor suggests that the “coating is microporous and infused with nano-hydroxyapatite (nHAp),” grounding vague concepts in the seed to more concrete concepts that the naive model could not even define if asked about them directly. The baseline and prompted naive refinements are also lifted by the mentor, but they do not provide much to build on. As a consequence, there is only so much the mentor can do with generic suggestions. Appendix G has more examples.

Results are different for the topic of neuroevolution. Here, the approaches are largely indistinguishable, i.e. there is no significant difference in judge scores (Figure 4a) or embedding space (Figure 4b). The problem is first that the base nanochat model does not know much about the niche topic of neuroevolution to begin with, so the difference among them is relatively small. Second, the model’s 1B capacity limits its ability to engage with complex topics in general, damping the possible advantage of being naive in such an area. All 1B variants in this domain tend to give generic machine learning advice that the mentor does its best to apply to neuroevolution, occasionally resulting in useful insights, but typically not pushing innovation.

Overall, the 1B true naive model tests hint at the potential for naive networks to provide seeds for alternative ideas different from what base networks would normally offer, but the limited intelligence at 1B constrains the breadth of domains where this potential can be applied.

## 5.2 Forgetting for Naive 7B Models

The limited capabilities of the 1B model in the neuroevolution domain imply that larger models may be necessary to ideate about more complex topics. However, as models get larger, pretraining time increases dramatically. Therefore, for 7B models, the forgetting method serves as the practical alternative to training from scratch. The increased capabilities of the larger model also motivate trying a wider range of topics: dental hygiene, neuroevolution, vaccination, and backpropagation.

Across all domains except backpropagation, the forgetting naive models produce more novel seed ideas that are refined to even more novel final proposals $( p < 0 . 0 0 1$ for seeds and refinements versus other methods, Figure 5a-c). Utility remains similar among seeds (except in a few cases, e.g. 7B Dental naive seeds are significantly higher, $p < 0 . 0 1 )$ , but when refined by the mentor the naive ideas also have significantly higher utility (at least $p < 0 . 0 1 $ ) except for the dental baseline.

Embedding distances generally show significant separation between baseline and naive models for seeds $( p < 0 . 0 5$ , see Figure 5d for examples) but only some of the refinements have significant embedding differences $( \mathbf { e . g . } p < 0 . 0 1$ for dental) over all prompts. This standardizing of representations through refinement was noted in the 1B model, but here the effect is more prominent. Despite that, qualitative analysis of word likelihood (Figure 6) and the judge scores strongly imply that the naive model is exploring unique ideas, such as in the 7B neuroevolution naive model (Figure 6a). That model focuses on adjacent fields like “Bayesian” or seemingly off-topic ideas like “spectral,”

→  
→  
![](images/38f1612c93def9b55b24416fb79bec006ea0bf8276200ee04f5e17c3b5986e80.jpg)  
(a) Baseline vs. Naive Differential Word Usage  
(b) Baseline vs. Prompted Differential Word Usage

Figure 3 Differential Word Usage in 1B Model Refinements. This figure shows the distinctive terms used in each model’s refinements for the given pair ranked by the z-scored weighted log-odds ratio with significance at $\vert z \vert > 1 . 9 6$ . Shared topic vocabulary is shrunk toward zero, so the listed terms reflect what differs between groups. The naive refinements focus on the methods (cleaning, wearable), while the baseline model sticks to standard dental cleaning implements. Comparing the prompted naive model to the baseline, most common cleaning implements (e.g. toothbrush) do not appear (except floss, which did not appear in (a) either) and are thus heavily shared between both the prompted and baseline models. All such comparisons are in Appendix F.

![](images/6fcf2ad29d54c1c8ce4e94b2f48dd03ead2aab73b3b38fc7c45b0633145d2b45.jpg)

![](images/b5496a998cd804042172e78bf4136539b7cb7ee33d101984d4fba3716a2f6483.jpg)

Prompt 2  
![](images/a100d9851e960462db343204df17d96ebfa7dad46a320ff6072d80e7b4fd2e20.jpg)

Prompt 3  
![](images/81a6c716acde5afbd5d01bf4c06c18a640aff9ced3f21262efcb48f1e8a0a0bc.jpg)

(a) Neuroevolution Judge Scores  
![](images/f53d0f6c2f4df7ec576a84b16fb0f23bdc9ad963139652ed2868fbe94daae420.jpg)  
(b) Neuroevolution Response Embeddings  
Figure 4 1B naive models do not work for complex topics. Results are presented in the same format as in Figure 2. For more complex topics like neuroevolution here, all the methods end up producing similarly rated (a) generic ideas that map similarly in embedding space after refinement (b). A stronger naive model allows the topic to be explored more effectively (later shown in Figure 5b). Bars show mean score across all prompts; error bars show 95% CI.

![](images/bc3641ca754c501ef7a5a1211d7b530e54ded03d2090bd9572d0d4ad03cdb100.jpg)

![](images/2e2dd83e48d1b271b526cbb1319c97b29e5e19feb4e8cbeb03cd297897c3ca3d.jpg)  
(a) Dental Hygiene Judge Scores

![](images/6c9c73e8a556ec209d99a7fdcb30eddbf1aedbd2cd05ec4b534594f570f6a255.jpg)  
(b) Neuroevolution Judge Scores

![](images/cc1d9c46668493e1ff67fdabf8449a2c5cd9bd96d35b4938d77242b8ac26173e.jpg)  
(c) Vaccination Judge Scores  
(d) Per-Topic Embeddings

Figure 5 7B models made to forget a topic outperform baselines on a variety of topics. Results are reported in the same format as in Figure 2. At this larger model size, naive forgetting seeds and refinements generate more novel and useful solutions with competitive feasibility scores (a-c). The naive seeds also tend to map to different parts of the embedding space compared to the baseline models, but the overlap increases for refinements as the mentor grounds them (d). See Appendix H for all embeddings. Bars show mean scores across all prompts; error bars show 95% CI.

though it is possible that the embedding models have difficulty separating nuance within specific topics. Additionally, the naive seeds and responses are consistently further from the baseline model than the prompted seeds are.

In contrast to these positive results for the naive (from forgetting) 7B networks, the results in the backpropagation domain at 7B look much like the neuroevolution results at 1B, with no significant difference in scores for seeds or refinements (Figure 7a). The embeddings (Figure 7b) do, however, show significant differences $( p < 0 . 0 0 1 )$ ) in the seeds, so the naive model does produce different solutions, just not particularly good ones. Overall the implication is that an even larger naive mode may be required to be able to grapple effectively with complicated machine learning topics.

Qualitatively, responses at 7B differ from the 1B model responses in several ways. First, due to the model size, the results tend to be more expressive and more grounded in reality, especially for scientific topics. The forgetting-naive models also have not completely forgotten the topics; they still make reference to common items in the domains but will make nuanced changes that trigger the mentor to think about the problem in a new light. For example, a vaccine seed idea is a patch that monitors viral exposure levels and the mentor refined it into a “CRISPR-Cas12/13 biosensing lapel pin that logs exposure events to measure real-world vaccine efficacy and guide targeted immunizations.” The combination of the fuzzy nature of topic boundaries and the strength of the pre-training examples may make it difficult to completely “forget” a topic in this manner, but the forgetting approach achieves the desired result of novel outputs by approaching naivety through the degradation of prior knowledge. Appendix G shares more examples.

Overall, the trend is consistent: whether through training from scratch or through forgetting, naive models consistently exhibit significantly more novel proposals than both baseline models and baseline models prompted to be naive. These results are generally preserved through mentor refinement, which also brings up feasibility and utility scores, though the novelty advantage may sometimes narrow (while remaining significant). At the same time, small naive models lose their advantage when the topic is too complex for their general level of competence. Nevertheless, the overall trend suggests the breadth of benefit of naive models should only grow as larger models are made naive.

## 6 Discussion

The argument is rarely made in machine learning that knowing less, having less organized understanding, or being deprived of relevant data can be an advantage, especially when the deficit is precisely in the area being evaluated. Indeed, in most problems and benchmarks, being naive or ignorant is likely a significant disadvantage. However, results with 1B and especially 7B models in this paper hint that under the right conditions problems involving creativity may be an exception. It might sometimes be useful for creative purposes to know less about the area you are going into.

Furthermore, it appears difficult to fake being naive; simply asking the 1B or 7B baseline model to take a naive perspective fails to compete with a model that went through a genuine forgetting process. This result makes intuitive sense; if having fresh eyes really does confer a creative benefit, then if it could work, many a one-hit wonder would simply pretend to be starting over again, but it clearly is not that easy.

The hint that knowing less can be advantageous cuts against longstanding assumptions in the field and undercuts the “knowing about everything” large model archetype that currently dominates both frontier inspiration and discourse. Yet while this less-is-more insight may initially seem surprising, it aligns with reasonable and longstanding intuitions about the value of fresh eyes (Azoulay et al., 2019). Unchained from dogma and freed from entrenched assumptions, it makes sense that a newcomer can pave a new path forward.

However, that newcomer must still be generally intelligent. One potential point of confusion would be to conflate being naive in a particular area with being generally less capable. Clearly overall diminished faculties are not the same advantage as coming to a field with fresh eyes and an otherwise vibrant mind, like a promising young student or novel entrant to a field.

This critical yet subtle distinction between a naive but gifted mind versus a generally impaired mind highlights why it was critical to perform baseline comparisons with models of the same overall capacity. That is, the naive (from forgetting) 7B model was compared to other 7B baselines (a regular 7B model and one asked to “pretend” to be naive). Even with a mentor, it is important to control for the capacity of the naive LLM because a mentor network that is genuinely helping another network refine its own ideas is intrinsically constrained by the capacity or intelligence of the network it is assisting.

![](images/b1e90970241d9197c7e541840ee4e42b429c6876a072e1025b8834ebe533172a.jpg)  
(a) Baseline vs. Naive Differential Word Usage for (b) Baseline vs. Prompted Differential Word Usage Neuroevolution for Vaccines  
Figure 6 Differential Word Usage in 7B Model Refinements. Results are reported in the same format as Figure 3. The 7B naive forgetting models follow the trend in the 1B models of focusing on fewer in-topic words and instead describe more about methods or adjacent approaches when compared to the baseline and prompted methods. For all such comparisons see Appendix F.

![](images/b3d206a6286fd793ca15bb5e1f4574d19bcbc9cf76c13213f76d70568f79a3e2.jpg)  
(a) Backpropagation Judge Scores

![](images/66b1a3ad7ae8519331de5055a83d394c205056421b546e4e16f495f004d477d1.jpg)  
(b) Backpropagation Response Embeddings  
Figure 7 7B naive models fail to gain an advantage in the backpropagation domain. The same format appears here as in Figure 2. Much like the 1B models failing to provide good seeds to neuroevolution, the 7B models struggle with backpropagation, with no significant difference in judge scoring (a) and all responses map to a similar embedding space after refinement (b). Bars show mean score across all prompts; error bars show 95% CI.

The nature of conventional LLM training makes it difficult to straightforwardly probe the implications of fresh eyes in a particular area. Training an entire LLM from scratch with one field of knowledge cut from its dataset is prohibitive (which is why we only attempted it at the 1B level) and we struggled to apply machine unlearning techniques to mature networks for this purpose. The forgetting method we introduce in this paper, which intentionally capitalizes on catastrophic forgetting, may not be the last word on obtaining naive LLMs, but it works well enough to suggest their intriguing potential.

In fact, the most important contribution of this work is likely not the method of forgetting, but rather the provocation of results suggesting that knowing less about something can be an advantage. That insight is likely a stepping stone to many novel ideas and experiments later to be conceived.

The mentor networks add a corollary to the main insight: the relative advantage of fresh eyes can be amplified with assistance. Note that the mentor in this paper is not a clean metaphor for mentors in real life, because it is both significantly more capable in general (being a frontier model versus the 7B model mentee) and also an expert in the area of ignorance of the naive LLM. Human apprenticementor relationships are likely to differ in the sense that the mentor is not necessarily vastly more intelligent in general. However, the results with mentors still help to highlight the fact that what might start out looking like only a subtle advantage from fresh eyes can end up greatly amplified under the right guidance.

A natural and intriguing question for the future is what a naive frontier-level LLM might be like. What would 1T or larger networks be capable of imagining given a blank slate in a particular area? Achieving such networks takes significantly more compute than 7B experiments in this paper, and also might require careful elaboration and calibration of the forgetting technique developed here that worked on much smaller networks. In the end, the best directions to take the central insight are unpredictable, but our hope is that the hint of potential here will inspire investigations in the future that heretofore would have been dismissed as unprincipled; one can no longer claim with full confidence that more data and more knowledge is always better, even when the deficit is in the domain of interest itself.

## 7 Conclusion

Purposefully restricting an LLM’s knowledge of a topic can mimic a human coming to a field with “fresh eyes,” unencumbered by existing bias or ways of approaching a problem. The naive models in this paper regularly outperformed baseline and naive prompted models in novelty and utility metrics, and embedding analysis showed that naive approaches produce a unique type of novelty. Pairing these naive models with a strong mentor generally boosted all aspects of proposals, including elevating feasibility to match baseline approaches. Contrary to scaling laws, this paper demonstrates a case where less knowledge can actually be more powerful, which merits future explorations across greater model sizes.

## References

Noga Alon, Thomas F. Bloom, W. T. Gowers, Daniel Litt, Will Sawin, Arul Shankar, Jacob Tsimerman, Victor Wang, and Melanie Matchett Wood. Remarks on the disproof of the unit distance conjecture, 2026.

Marti J. Anderson. A new method for non-parametric multivariate analysis of variance. Austral Ecology, 26(1):32–46, 2001. doi: 10.1111/j.1442-9993.2001.01070.pp.x. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1442-9993. 2001.01070.pp.x.

Anthropic. Claude opus 4.5 system card. System card, November 2025. URL https: //www.anthropic.com/claude-opus-4-5-system-card. API model snapshot claude-opus-4-5-20251101. Accessed 2026-09-10.

Pierre Azoulay, Christian Fons-Rosen, and Joshua S. Graff Zivin. Does science advance one funeral at a time? American Economic Review, 109(8):2889–2920, August 2019. ISSN 0002-8282. doi: 10.1257/aer.20161574.

Wacha Bounliphone, Eugene Belilovsky, Matthew B. Blaschko, Ioannis Antonoglou, and Arthur Gretton. A test of relative similarity for model selection in generative models. In International Conference on Learning Representations, 2016. URL https://arxiv.org/abs/1511. 04581.

Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, and Dmitrii Krasheninnikov. Understanding (un)reliability of steering vectors in language models, 2025. URL https://arxiv.org/abs/2505.22637. ICLR 2025 workshop paper.

Tara Chari and Lior Pachter. The specious art of single-cell genomics. PLOS Computational Biology, 19(8):e1011288, 2023.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https://arxiv.org/abs/1803.05457.

Haochuan Cui, Yiling Lin, Lingfei Wu, and James A. Evans. Aging and the narrowing of scientific innovation. Science, 392(6798):588–591, 2026. doi: 10.1126/science.ady8732.

Omkar Dige, Diljot Arneja, Tsz Fung Yau, Qixuan Zhang, Mohammad Bolandraftar, Xiaodan Zhu, and Faiza Khan Khattak. Can machine unlearning reduce social bias in language models? In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 954–969, Miami, Florida, US, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-industry.71. URL https://aclanthology.org/ 2024.emnlp-industry.71/.

Ronen Eldan and Mark Russinovich. Who’s harry potter? approximate unlearning in llms, 2023. URL https://arxiv.org/abs/2310.02238.

Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2426–2436, October 2023.

Google DeepMind. Gemini 3.1 pro model card. Model card, February 2026. URL https: //deepmind.google/models/model-cards/gemini-3-1-pro/. API model string gemini-3.1-pro-preview. Accessed 2026-09-08.

Alfredo Guevara, Alexandru Lupsasca, David Skinner, Andrew Strominger, and Kevin Weil. Singleminus gluon tree amplitudes are nonzero, 2026.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= d7KBjmI3GmQ.

Michael Hla. Machina mirabilis. Blog post, March 2026. URL https://michaelhla. com/blog/machina-mirabilis.html. Model and code: https://github. com/michaelhla/gpt1900; weights: https://huggingface.co/collections mhla/gpt-1900. Accessed 2026-09-08.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol´ Vinyals, Jack Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems, volume 35, pp. 30016–30030. Curran Associates, Inc., 2022. doi: 10.52202/068431-2176. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html. arXiv:2203.15556.

Lu Hong and Scott E. Page. Groups of diverse problem solvers can outperform groups of high-ability problem solvers. Proceedings ofthe National Academy ofSciences, 101(46):16385–16389, 2004. ISSN 0027-8424. doi: 10.1073/pnas.0403723101.

Peter L. Jakab. Visions of a Flying Machine: The Wright Brothers and the Process of Invention. Smithsonian History of Aviation Series. Smithsonian Institution Press, Washington, D.C., 1990.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14389–14408, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.805. URL https://aclanthology.org/ 2023.acl-long.805/.

Lars Bo Jeppesen and Karim R. Lakhani. Marginality and problem-solving effectiveness in broadcast search. Organization Science, 21(5):1016–1033, October 2010. ISSN 1047-7039. doi: 10.1287/orsc.1090.0491.

Benjamin F. Jones. Age and great invention. The Review of Economics and Statistics, 92(1):1–14, 2010. doi: 10.1162/rest.2009.11724.

Mary Kaltenberg, Adam B. Jaffe, and Margie E. Lachman. Invention and the life course: Age dif ferences in patenting. Research Policy, 52(1):104629, 2023. doi: 10.1016/j.respol.2022.104629.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020. doi: 10.48550/arXiv.2001.08361. URL https:// arxiv.org/abs/2001.08361.

Andrej Karpathy. nanochat: The best chatgpt that \$100 can buy. GitHub repository, 2025. URL https://github.com/karpathy/nanochat. Commit f5a0ea4 (2026-01-08). Accessed 2026-09-08.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the effects of RLHF on LLM generalisation and diversity. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=PXD3FAVHJT. arXiv:2310.06452.

Thomas S. Kuhn. The Structure ofScientific Revolutions. University of Chicago Press, Chicago, IL, 4th, 50th anniversary edition, 2012. ISBN 978-0-226-45811-3. With an introductory essay by Ian Hacking. Originally published 1962.

Carolyn Lamb, Daniel G. Brown, and Charles L. A. Clarke. Evaluating computational creativity: An interdisciplinary tutorial. ACM Computing Surveys, 51(2):1–34, February 2018. ISSN 0360- 0300. doi: 10.1145/3167476. URL https://doi.org/10.1145/3167476.

Joel Lehman and Kenneth O. Stanley. Abandoning objectives: Evolution through the search for novelty alone. Evolutionary Computation, 19(2):189–223, 2011. doi: 10.1162/EVCO a 00025.

Fanfei Li, Jana Zeller, Manuel Prada-Corral, Thaddaus Wiedemer, Prasanna Mayilvahanan, Ryan¨ Cotterell, and Wieland Brendel. Littlelearner: Language models under pedagogically controlled knowledge exposure, 2026a. URL https://arxiv.org/abs/2608.13545. NeurIPS 2026 (oral); proceedings not yet published.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Gabriel Mukobi, Nathan Helm-Burger, Rassin Lababidi, Lennart Justen, Andrew Bo Liu, Michael Chen, Isabelle Barrass, Oliver Zhang, Xiaoyuan Zhu, Rishub Tamirisa, Bhrugu Bharathi, Ariel Herbert-Voss, Cort B Breuer, Andy Zou, Mantas Mazeika, Zifan Wang, Palash Oswal, Weiran Lin, Adam Alfred Hunt, Justin Tienken-Harder, Kevin Y. Shih, Kemper Talley, John Guan, Ian Steneker, David Campbell, Brad Jokubaitis, Steven Basart, Stephen Fitz, Ponnurangam Kumaraguru, Kallol Krishna Karmakar, Uday Tupakula, Vijay Varadharajan, Yan Shoshitaishvili, Jimmy Ba, Kevin M. Esvelt, Alexandr Wang, and Dan Hendrycks. The WMDP benchmark: Measuring and reducing malicious use

with unlearning. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 28525–28550. PMLR, 2024. URL https://proceedings.mlr.press/v235/li24bc.html. arXiv:2403.03218.

Zehan Li, Yuxuan Wang, Ali El Lahib, Ying-Jieh Xia, and Xinyu Pi. Simulated ignorance fails: A systematic study of LLM behaviors on forecasting problems before model knowledge cutoff. In Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-26), pp. 5837–5845, 2026b. doi: 10.24963/ijcai.2026/650. URL https://doi.org/10. 24963/ijcai.2026/650. Main Track; arXiv:2601.13717.

Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Yuguang Yao, Chris Yuhao Liu, Xiaojun Xu, Hang Li, Kush R. Varshney, Mohit Bansal, Sanmi Koyejo, and Yang Liu. Rethinking machine unlearning for large language models. Nature Machine Intelligence, 7(2):181–194, February 2025. doi: 10.1038/s42256-025-00985-0. URL https: //www.nature.com/articles/s42256-025-00985-0.

TaiMing Lu and Philipp Koehn. Learn and unlearn: Addressing misinformation in multilingual LLMs. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 10180–10195, Suzhou, China, November 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.516. URL https://aclanthology.org/ 2025.emnlp-main.516/.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE Transactions on Audio, Speech and Language Processing, 33:3776–3786, 2025. doi: 10.1109/TASLPRO.2025. 3606231. arXiv:2308.08747.

Marlene Lutz, Indira Sen, Georg Ahnert, Elisa Rogers, and Markus Strohmaier. The prompt makes the person(a): A systematic evaluation of sociodemographic persona prompting for large language models. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 23212– 23237, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8- 89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.1261. URL https://aclanthology. org/2025.findings-emnlp.1261/.

Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, volume 24, pp. 109–165. Elsevier, 1989.

Burt L Monroe, Michael P Colaresi, and Kevin M Quinn. Fightin’words: Lexical feature selection and evaluation for identifying the content of political conflict. Political Analysis, 16(4):372–403, 2008.

Charlan J. Nemeth. Differential contributions of majority and minority influence. Psychological Review, 93(1):23–32, 1986. ISSN 0033-295X. doi: 10.1037/0033-295X.93.1.23.

Charlan Jeanne Nemeth and Joel Wachtler. Creative problem solving as a result of majority vs minority influence. European Journal of Social Psychology, 13(1):45–55, 1983. doi: 10.1002/ ejsp.2420130103.

Kyle O’Brien, Stephen Casper, Quentin Anthony, Tomek Korbak, Robert Kirk, Xander Davies, Ishan Mishra, Geoffrey Irving, Yarin Gal, and Stella R Biderman. Deep ignorance: Filtering pretraining data builds tamper-resistant safeguards into open-weight llms. In International Conference on Learning Representations, volume 2026, pp. 35579–35633, 2026.

OpenAI. GPT-5.6 luna. Model documentation, OpenAI API, 2026a. URL https: //developers.openai.com/api/docs/models/gpt-5.6-luna. Model snapshot gpt-5.6-luna; knowledge cutoff 2026-02-16. Accessed 2026-09-08.

OpenAI. openai/math: Mathematical manuscripts and proof artifacts. GitHub repository, 2026b. URL https://github.com/openai/math. Accessed 2026-10-17.

Mikko Packalen and Jay Bhattacharya. Age and the trying out of new ideas. Journal of Human Capital, 13(2):341–373, 2019. doi: 10.1086/703160.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems, volume 37, pp. 68772– 68802, 2024. doi: 10.52202/079017-2197. arXiv:2404.13076.

Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, volume 37, pp. 30811–30849, 2024. doi: 10.52202/079017-0970. arXiv:2406.17557.

Justin K. Pugh, Lisa B. Soros, and Kenneth O. Stanley. Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI, 3:40, July 2016. ISSN 2296-9144. doi: 10.3389/frobt.2016.00040. URL https://doi.org/10.3389/frobt.2016.00040.

Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings ofthe 58th annual meeting of the association for computational linguistics, pp. 7237–7256, 2020.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering llama 2 via contrastive activation addition. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15504–15522, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.828. URL https://aclanthology.org/2024.acl-long.828/.

Rick Rubin. The Creative Act: A Way of Being. Penguin Press, 2023. ISBN 978-0593652886.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nature, 550(7676):354–359, October 2017. ISSN 0028- 0836. doi: 10.1038/nature24270. URL https://doi.org/10.1038/nature24270.

Kenneth O. Stanley, Jeff Clune, Joel Lehman, and Risto Miikkulainen. Designing neural networks through neuroevolution. Nature Machine Intelligence, 1(1):24–35, January 2019. ISSN 2522-5839. doi: 10.1038/s42256-018-0006-z. URL https://doi.org/10.1038/ s42256-018-0006-z.

Anirudh Subramanyam, Yuxin Chen, and Robert L. Grossman. Scaling laws revisited: Modeling the role of data quality in language model pretraining. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2510.03313.

Richard S. Sutton. The bitter lesson. Incomplete Ideas (blog), March 2019. URL http://www. incompleteideas.net/IncIdeas/BitterLesson.html. Accessed 2026-09-08.

Team Olmo. Olmo 3, 2025. URL https://arxiv.org/abs/2512.13961.

Abishek Thamma and Micha Heilbron. Human-like fleeting memory improves language learning but impairs reading time prediction in transformer language models. Transactions of the Associ ationfor Computational Linguistics, 14:877–892, 2026. doi: 10.1162/tacl.a.688.

George Tsoukalas, Anton Kovsharov, Sergey Shirobokov, Anja Surina, Moritz Firsching, Gergely Berczi, Francisco J. R. Ruiz, Arun Suggala, Adam Zsolt Wagner, Eric Wieser, Lei Yu, Aja Huang,´ Miklos Z. Horv´ ath, Andrew Ferraiuolo, Henryk Michalewski, Codrut Grosu, Edward Lockhart,´ Thomas Hubert, Matej Balog, Pushmeet Kohli, and Swarat Chaudhuri. Advancing mathematics research with AI-driven formal proof search. Science, 394(6820):234–239, 2026. doi: 10.1126/ science.aej2213. Preprint: arXiv:2605.22763.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering, 2024. URL https://arxiv.org/abs/2308.10248. Version 5, October 2024.

![](images/462a2b9c7a01124de7055a1fb77455d2dadf4b284a048b3cd5208003f4e833c3.jpg)

Brian Uzzi, Satyam Mukherjee, Michael Stringer, and Ben Jones. Atypical combinations and scientific impact. Science, 342(6157):468–472, October 2013. ISSN 0036-8075. doi: 10.1126/science. 1240474.

Teun van der Weij, Felix Hofstatter, Ollie Jaffe, Samuel F. Brown, and Francis Rhys Ward. AI¨ sandbagging: Language models can strategically underperform on evaluations. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2406. 07358.

Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Position: Will we run out of data? Limits of LLM scaling based on human-generated data. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 49523–49544. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/villalobos24a.html.

Martin Wattenberg, Fernanda Viegas, and Ian Johnson. How to use t-sne effectively.´ Distill, 1(10): e2, 2016.

Anna Wegmann, Marijn Schraagen, and Dong Nguyen. Same author or just same topic? towards content-independent style representations. In Proceedings of the 7th Workshop on Representation Learningfor NLP, pp. 249–268, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.repl4nlp-1.26. URL https://aclanthology.org/ 2022.repl4nlp-1.26/.

Orion Weller, Michael Boratko, Iftekhar Naim, and Jinhyuk Lee. On the theoretical limitations of embedding-based retrieval. In The Fourteenth International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2508.21038. arXiv:2508.21038.

Jennifer Wiley. Expertise as mental set: The effects of domain knowledge in creative problem solving. Memory and Cognition, 26(4):716–730, July 1998. ISSN 0090-502X. doi: 10.3758/ BF03211392.

Jiayi Zhang, Simon Yu, Derek Chong, Anthony Sicilia, Michael Tomz, Christopher D Manning, and Weiyan Shi. Verbalized sampling: How to mitigate mode collapse and unlock LLM diversity. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 160073–160152. PMLR, July 2026. URL https://proceedings.mlr.press/v306/zhang26hy.html. arXiv:2510.01171.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models, 2025. URL https: //arxiv.org/abs/2506.05176.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623, 2023. doi: 10.52202/075280-2020. Datasets and Benchmarks Track; arXiv:2306.05685.

## A Mentor Prompting

The mentor model is asked to expand seed ideas from other models. The seed ideas are expected to be inspirational for the mentor and may be impractical or be missing information; the mentor must fill in these gaps. The mentor is presented with the seed idea prefixed with the following prompt that is designed to enforce faithfulness to the seed idea.

{seed\_text}

- Stay faithful to the seed: develop its core concept, not a different   
invention.

\- If the seed lists multiple ideas, pick exactly ONE - the most unique, surprising, or interesting FROM the seed - and develop only that. Do not default to the safest or most conventional option.

\- If the seed is generic advice, a how-to checklist, or only names existing products with no new invention, do NOT invent an unrelated commercial product. State briefly that the seed lacks a concrete invention and refine only whatever specific idea (if any) is actually present.

- Elaborate mechanism, form, and use for the selected idea - including   
speculative or fantastical concepts. Add plausible engineering detail   
where the seed is vague; do not substitute a standard off-the-shelf   
product or method unless the seed proposed it.

## B Training Configurations

Models trained in this paper were chosen because their pipelines are fully open source and the only change to the pipelines is the data filtering described in Section 3. Additional details are reproduced here for convenience.

## B.1 1B nanochat

The nanochat (Karpathy, 2025) model is trained with depth 26: a 1.08B-parameter decoder-only Transformer (26 layers, d model 1664, 13 heads, 2048-token context, 65,536-token BPE vocab) with RoPE, QK-norm, parameter-free RMSNorm, untied embeddings, and ReLU² MLPs. Pretraining uses FineWeb-EDU (100BT shuffled) for 21.6B tokens, i.e. the Chinchilla-optimal 20 tokens per parameter, followed by midtraining on 848K conversations (SmolTalk, MMLU auxiliary train, GSM8K, spelling tasks, synthetic identity data) and supervised finetuning on 23K conversations with loss on assistant tokens only. Token counts are for baseline models and are reduced for naive training. The optional RL stage is not run. All hyperparameters are nanochat defaults (commit f5a0ea4) except model depth (20 → 26), the pretraining shard count (450), the per-device batch size (32 → 16, for memory), and the tokens-to-parameters ratio (set to 20).

## B.2 7B Olmo

We initialized from the publicly released Olmo 3 7B base model (allenai/Olmo-3-1025-7B) and performed supervised fine-tuning (SFT) only, with no preference tuning or reinforcement learning. The base model was pretrained by AI2 in three stages: general pretraining on the Dolma 3 corpus (5.9T tokens, primarily web data plus academic PDFs), midtraining on the 100B-token Dolma 3 Dolmino Mix targeting math, code, and reasoning, and long-context extension on the Dolma 3 Longmino Mix. SFT was based on AI2’s Dolci-Instruct-SFT mixture (about 2.15M samples) from the Olmo 3 post-training release.

## C Topic Prompts

For each of the four topics, three prompts were selected to elicit creative and productive responses. The prompts are shown below:

• Dental

– Scientists have discovered that debris on teeth can cause damage to them. Can you help me invent a tool to help people clean their teeth?

– My teeth are dirty. Can you help me invent something to clean my teeth?

– Can you help me invent something to clean between my teeth?

## • Backpropagation

– I have a neural network and I want to teach it to recognize images. Can you help me invent a method to train it?

– My neural network is making wrong predictions. Can you help me invent a way to adjust its weights so it makes better predictions?

– My neural network is a chain of mathematical functions. Can you help me invent a way to calculate how sensitive the final output is to changes in the early layers?

## • Vaccines

– Scientists have discovered that people can get sick from infectious diseases. Can you help me invent a way to protect people from getting sick?

– I want to train someone’s immune system to fight off a disease before they get it. Can you help me invent a method to do this?

– I’d like to prevent a disease from spreading through a population. Can you help me invent a way to do this?

## • Neuroevolution

– I want to train a neural network but I don’t want to use gradients. Can you help me invent a way to train a network without using gradients?

– I need to find a good neural network architecture for my problem. Can you help me invent a way to find a good network architecture?

– I’d like to generate many different neural networks that each solve my problem in their own way. Can you help me invent a way to generate diverse neural network solutions?

## D Topic Filters

Filtering depends on simple substring matching for most terms, but for acronyms and smaller terms word boundaries are checked. The English n-grams are shown below, but automated translations to other dominant languages in the training sets are also filtered.

Dental: toothbrush, toothpaste, dental floss, flossing, mouthwash, water flosser, interdental brush, tongue scraper, dental hygiene, oral hygiene, plaque removal, tartar, tooth decay, fluoride, gum care, cavity prevention, enamel protection, dentist, dental professional, waterpik, listerine, colgate, sensodyne, crest, aquafresh, brush teeth, floss your teeth, tongue scraping, oil pulling, denture cleaning, scaling and polishing.

Neuroevolution: neuroevolution, neuro-evolution, NEAT, HyperNEAT, CPPN, TWEANN, augmenting topologies, topology and weight evolving, evolving neural network topology, compositional pattern producing network, indirect encoding, speciation, innovation number, historical markings, compatibility distance, structural mutation, node gene, connection gene, CoSyNE, enforced subpopulations, cooperative neuroevolution, evolve neural networks, deep neuroevolution, rtNEAT, ES-HyperNEAT, neat-python, sharpneat, tensorneat.

Backpropagation: backpropagation, backprop, back-propagation, back-prop, backward pass through the network, backward through the network, backpropagating, backpropagated, reverse-mode autodiff, reverse mode automatic differentiation.

Vaccines: vaccine, vaccination, vaccinate, immunization, immunisation, immunize, immunise, flu shot, booster shot, booster dose, herd immunity, inoculation, inoculated, vax.

## E Judge Rubric

## Novelty

• 1–3: Not a concrete invention; generic advice; or the selected item exactly matches a wellknown existing technique.

• 4: Small, obvious tweak to a familiar approach (e.g., adding logging to a standard algorithm).

• 5: Names a specific device or mechanism that differs from the default—even if components exist elsewhere. Combinatorial: basic feature addition.

• 6: Clever or unexpected combination of existing technologies not widely used; would mildly surprise a domain expert. Combinatorial: interestingfeature mashup.

• 7: Distinct twist on a known approach—an unusual mechanism or feature combination that is still recognizably within an existing product category. Combinatorial: sophisticated integration.

• 8: Conceptual leap—proposes a new kind of physical interaction, form factor, or paradigm that doesn’t map cleanly to existing product categories. Examples:

– A device that adapts to its environment and changes behavior/shape

– A mechanism that harvests energy from its operating context (motion, heat, vibration)

– A new physical relationship between system components (self-assembly, emergent coordination)

– An autonomous system that navigates or self-organizes

• 9: A fundamentally new mechanism of action rarely seen even in research—a novel physical principle, unexpected energy source, or new theoretical framework applied to the domain.

• 10: Paradigm-shifting—redefines the field itself. Would overturn how experts think about the problem; no close analog anywhere.

## Utility

• 8–10: Clear potential advantage over existing approaches—faster, cheaper, more effective, reaches cases current methods can’t, or solves a limitation of current approaches. The proposal identifies why this would be better.

• 6–7: Comparable to existing approaches with some potential advantages in specific scenarios. May be better for certain use cases but not a clear overall improvement.

• 4–5: Would work but offers no clear advantage over what already exists. A lateral move, not an improvement. “Yet another X” without a compelling reason to prefer it.

• 1–3: Likely worse than existing approaches, doesn’t actually solve the problem, or is offtopic.

## Feasibility

• 1–3: Vague direction only. No concrete approach specified. “Improve the process”, “use AI”, “make it better”. OR contains clear physics violations or internal contradictions.

• 4–5: Named approach without specifics. Identifies a technique or method but provides no implementation details. “Use backpropagation”, “use a CNN”, “follow standard protocols”. An expert could fill in the blanks, but the text itself doesn’t.

• 6–7: Approach with some specifics. Provides meaningful implementation details but gaps remain. Clear enough to start, but not a complete design.

• 8–10: Detailed design. Specifies architecture, parameters, materials, procedures, or mechanisms in enough detail that implementation could begin. Acknowledges challenges and proposes solutions.

## Faithfulness (Refinements Only)

• 1–2: Complete substitution—the refinement describes an invention or approach that has no meaningful connection to the seed. The mentor ignored the seed entirely and invented something unrelated.

• 3–4: Partial substitution—the refinement loosely references the seed but the core mechanism or concept is different. The mentor may have cherry-picked a minor detail while replacing the main idea.

• 5–6: Weak alignment—the refinement addresses the same general problem domain as the seed but takes a noticeably different approach. Some connection exists but the refinement diverges significantly.

• 7–8: Good alignment—the refinement clearly develops the seed’s core concept. The mechanism or approach is recognizably derived from the seed, with added detail, refinement, or engineering specifics.

• 9–10: Excellent alignment—the refinement is a faithful, detailed elaboration of the seed’s specific idea. The core concept is preserved and enhanced without substitution.

## F Differential Word Usages

This section shows differential word usages for all model pairs across topics and sizes. See Figures 8-13.

![](images/d7f9ff2a72b3c4f14412249c993566460709e4d879b731fcea517dfbe9b015ea.jpg)  
(a) Naive vs. Baseline Differential Word Usage

![](images/37ea30d09ecf48748ec85480ca7e45b251dc20587aad0f7aba525f08498e13c4.jpg)  
(b) Naive vs. Prompted Differential Word Usage

![](images/93f7ca2092f0d71c20cfd032c26a0d50e39e197559477f9f9b0ce2eb4a3727d9.jpg)  
(c) Baseline vs. Prompted Differential Word Usage

Figure 8 1B Dental: distinctive terms in refined outputs for each pair of models.  
![](images/131ff0a6f18126bbe87c79d7157ccc410d65b9e380b1e03f19e8bb4723f981e3.jpg)  
(a) Naive vs. Baseline Differential Word Usage

![](images/ce3931e0333e709f21dec303741c62361f9ab587f0dc280d473d1c3efb77534c.jpg)  
(b) Naive vs. Prompted Differential Word Usage

![](images/f6834583309f3f4560a54acc594f8a4fb6a2fff4281c16a6e54715d3554cd5ac.jpg)  
(c) Baseline vs. Prompted Differential Word Usage

Figure 9 1B Neuroevolution: distinctive terms in refined outputs for each pair of models.

![](images/5bf48bb48cbc26b16ba62830a8987cca434d868f90d19539aceacfd03cd357d8.jpg)  
(a) Naive vs. Baseline Differential Word Usage

![](images/b1bbdd9f39db7fe5b9d9a6ae0a3d2c8bdfead3a2a99b08cce595478b203a6b0d.jpg)  
(b) Naive vs. Prompted Differential Word Usage

![](images/abea1ced0caef0ee30ee991756212271b7904ff6cfd9b70f4c75bfd77a6c93d4.jpg)  
(c) Baseline vs. Prompted Differential Word Usage  
Figure 10 7B Dental: distinctive terms in refined outputs for each pair of models.

![](images/7a9b1ecd2cd394122c1d5394341f89a88feee0b6fab99bad4c991aed39291190.jpg)  
(a) Naive vs. Baseline Differential Word Usage

![](images/f0ec464ac9524f522b805225faacce9f17504ef0e3f6551152de60bbac6d27eb.jpg)  
(b) Naive vs. Prompted Differential Word Usage

![](images/8ac806fdf914b8b327bc6771327e7a37cc379fa0d254b1e9d9c177398c9d0085.jpg)  
(c) Baseline vs. Prompted Differential Word Usage  
Figure 11 7B Neuroevolution: distinctive terms in refined outputs for each pair of models.

![](images/a66ba5aa56aa724026be5d52b9f19c995c8549012aa8d6b603ba45f9744884bf.jpg)  
(a) Naive vs. Baseline Differential Word Usage

![](images/80e149011fe289c25c19456f49d1494a9426a528b7d6e53cd3cb95ad40c3860a.jpg)  
(b) Naive vs. Prompted Differential Word Usage

![](images/0dab3b0d2637a307846ec901718d1b2d2ce2bf7068ca2c67a038d560509776f3.jpg)  
(c) Baseline vs. Prompted Differential Word Usage  
Figure 12 7B Vaccines: distinctive terms in refined outputs for each pair of models.

![](images/5aa36f2d461f330db1782d9fefcb34ab2dac56e474de2085221caf6f3fa90b4b.jpg)  
(a) Naive vs. Baseline Differential Word Usage

![](images/99e524e262ac48e39b8f82bbe6ecb25643d73c2bceee568084e706d61af358d7.jpg)  
(b) Naive vs. Prompted Differential Word Usage

![](images/1dc224494c3525925993222440c3bea28119bed1d44a7ef9b2de76b6321d0a52.jpg)  
(c) Baseline vs. Prompted Differential Word Usage  
Figure 13 7B Backpropagation: distinctive terms in refined outputs for each pair of models.

## G Example Seeds and Refinements

This section contains representative samples of the seeds and refinements by the naive and baseline models at 1B and 7B sizes.

Table 1 1B dental: Example seeds and their frontier-model expansions (verbatim; [. . . ] marks a cut). <sup>†</sup>Frontier model noted the seed had no concrete invention.
<table><tr><td>Model</td><td>Seed Idea</td><td>Refinement</td></tr><tr><td>Naive</td><td>“Imagine a smart, self-cleaning surface that&#x27;s integrated into your mouth. This surface, which we can call &quot;TongueCycle,&quot; would be made up of tiny, tooth-like projections that can be controlled by your tongue. As you chew and speak, these projections would push and pull against a conveyor belt-like system, removing dirt, food particles, and even a [...]”</td><td>TongueCycle. “The “TongueCycle&quot; is a speculative but mechanically plausible intraoral device designed to provide continuous, automated interdental cleaning and oral microbiome management. Worn similarly to a custom-fitted lingual retainer (sitting discreetly behind the lower or upper teeth), the device utilizes kinetic energy from the user&#x27;s tongue, speech, and mastication to drive a micro-mechanical “conveyor&quot; system.</td></tr><tr><td>Naive</td><td>“Imagine a device that not only collects debris but also uses it to stimulate the growth of new teeth. This tool, which we can call “DENTICLE,&quot; would be designed to capture and break down organic debris, such as gum, gum tissue, and even roots of dead teeth, using a precision-crafted, handheld tool. [.. . ]&quot;</td><td>[...]” The DENTICLE: Autologous Bio-Scaffold Harvester &amp; Applicator. “The DENTICLE is a handheld, regenerative dental tool that bridges the gap between daily oral hygiene and advanced tissue engineering. Instead of merely discarding oral debris, the device harvests autologous (self-derived) biological matter from the user&#x27;s mouth—specifically shed epithelial cells, periodontal ligament fragments, and organic root debris. [.. . ]”</td></tr><tr><td>Naive</td><td>“Imagine a device that not only cleans your teeth but also monitors your health and provides personalized recommendations. This device, which we can call “Dash,&quot; can detect the type of cleaning needed, the materials used, and even the environment you&#x27;re in, adjusting its cleaning routine accordingly. [...]”</td><td>Project Dash: The Edible Smart-Chew Dental Monitor. &quot;Dash is a bio-compatible, fully edible smart lozenge/chew designed to replace traditional midday brushing or gum. It acts as an autonomous in-vivo sensor and mechanical cleaner. As the user chews Dash, it maps the oral microbiome, physically scrubs the teeth, dynamically releases cleaning agents only where needed, and transmits health data to a smartphone before</td></tr><tr><td>Baseline†</td><td>“A popular and effective method for cleaning between teeth is using an interdental floss or a interdental brush. These tools gently slide plaque, and food particles that can accumulate. You can place them between each tooth and along the gumline, and they&#x27;ll help keep your teeth healthy. [.. . ]”</td><td>[...]” The Xero-Irrigator: A Dry-Mouth Optimized Water Flosser. &quot;The Xero-Irrigator refines the standard water between each tooth, gently removing bacteria, flosser into a gentle, hydrating tool that cleans between teeth while actively treating the user&#x27;s dry mouth condition.&quot;</td></tr><tr><td>Prompted naive†</td><td>“There are several things you can do to keep your teeth healthy between them. One thing you can try is to use a dental floss or interdental floss. This tool can be used to remove food particles and plaque from between your teeth.&quot;</td><td>Variable-Profile Interdental Floss Matrix. “Taking the seed&#x27;s concept of an “interdental floss,&quot;this proposal refines the tool into a continuous, variable-profile filament that combines the tight-contact insertion of standard floss with the volumetric cleaning power of an interdental brush.&quot;</td></tr></table>

Table 2 1B neuroevolution: Example seeds and their frontier-model expansions (verbatim; [. . . ] marks a cut).
<table><tr><td>Model</td><td>Seed Idea</td><td>Refinement</td></tr><tr><td>Naive</td><td>“One approach you can use is by using a data augmentation technique. This technique involves adding random perturbations to your data to create new training data. By randomly varying the input values, you can create new instances that are similar to the original dataset. [... ]”</td><td>Evolutionary Feature Perturbation (EFP). “The seed proposes a fascinating paradigm shift: rather than using evolutionary algorithms or gradient descent to directly update a neural network&#x27;s weights, we apply random perturbations to the data (specifically, the intermediate features) to find better representations, entirely bypassing the overhead of backpropagation and gradient calculation.&quot;</td></tr><tr><td>Baseline</td><td>“One way to train a neural network without using gradients is by using a technique called &quot;Dropout.&quot; Dropout randomizes the loss function of the network, so that if you use the same loss function on two different sets of data, the network will still learn the same patterns and relationships. [.. . ]”</td><td>Stochastic Fitness Dropout (SFD) for Neuroevolution. “Taking the seed&#x27;s idea, Stochastic Fitness Dropout (SFD) solves this by dynamically randomizing and masking components of the fitness function across different data batches. Because the evolutionary algorithm cannot rely on a single, static loss metric to select the best networks, the surviving networks are forced to develop robust, generalized internal</td></tr><tr><td>Prompted naive</td><td>“Let&#x27;s consider a few approaches. One way is by using a simple backpropagation (learning rate) method. This involves setting the network&#x27;s weights and biases to fit the data and the desired output, and then updating these weights and biases based on the model&#x27;s of a static activation function, networks in the performance on unseen data. [.. . ]”</td><td>representations (the &quot;most important [... ]” Ontogenetic Threshold Annealing (OTA). &quot;Ontogenetic Threshold Annealing (OTA) takes the seed&#x27;s idea of a “lower threshold ReLU” used in “early stages&quot; and turns it into a dynamic, evolutionary curriculum. Instead population undergo an “ontogenetic&quot; (developmental) shift where the activation threshold starts extremely low (allowing negative signals to pass) and gradually</td></tr></table>

Table 3 7B dental: example seeds and their frontier-model expansions (verbatim; [. . . ] marks a cut). <sup>†</sup>Frontier model noted the seed had no concrete invention.
<table><tr><td>Model</td><td>Seed Idea</td><td>Refinement</td></tr><tr><td>Naive</td><td>“A miniaturized robotic armada that can be controlled to maneuver around teeth of any size. These miniature robotic arms could possibly be controlled by air or water pressure.&quot;</td><td>The Hydro-Armada Micro-Robotic Cleansing System. “The Hydro-Armada takes the seed&#x27;s concept of a “miniaturized robotic armada&quot; and grounds it in the emerging field of microfluidic soft robotics. Traditional toothbrushes rely on static bristles dragging across surfaces. The Hydro-Armada system replaces bristles with thousands of microscopic, hydraulically actuated</td></tr><tr><td>Naive</td><td>“A stringless floss that can be controlled to move around teeth and remove plaque.&quot;</td><td>elastomeric “arms&quot; housed inside a mouthguard. [.. . ]” FluxFlossTM (Dynamic Magnetic-Particle Flossing System). &quot;FluxFloss is a handheld smart dental device that replaces traditional nylon or Teflon string with a magnetically controlled, dynamic “wire&quot; made of a biocompatible nano-fluid. It provides the exact mechanical scraping action of traditional floss, but because the “string&quot; is formed by a magnetic field, it can instantly</td></tr><tr><td>Naive</td><td>“A miniature robotic arm that is controlled by a sensor in your saliva. The arm would be able to reach all areas of your mouth to remove plaque and food debris.&quot;</td><td>break apart and reform to pass through [.. . ]&quot; The Auto-Chemotactic Oral Manipulator (ACOM). &quot;The ACOM is a wearable, intraoral device featuring a soft-robotic continuum arm that autonomously seeks out and removes plaque and food debris. Rather than relying on manual brushing, the device is guided by a real-time chemical analysis of the user&#x27;s saliva, allowing it to target the exact locations of bacterial activity and trapped</td></tr><tr><td>Baseline†</td><td>“Water flosser - A water flosser is a device that uses a stream of water to clean the spaces between the teeth.&quot;</td><td>food.&quot; The Water Flosser (Oral Irrigator). “A device that utilizes a highly pressurized, pulsating stream of water to shear away plaque biofilm, dislodge food particles, and flush out bacteria from interdental spaces (between the teeth) and subgingival pockets</td></tr><tr><td>Prompted naive</td><td>“Use a clean cloth or paper towel to gently wipe away any visible debris from your teeth with gentle circular motions until all visible debris has been removed from your teeth.&quot;</td><td>(below the gumline).&quot; The “Circu-Wipe&quot; Dental Textile Sleeve. “Instead of a standard, unhygienic paper towel or bulky washcloth, the Circu-Wipe is a disposable, biodegradable finger-cot made from a highly engineered dental textile. The Exterior (Cleaning Surface): Constructed from a tightly woven, medical-grade bamboo microfiber. The fabric features a patented “micro-loop” texture. [.. . ]”</td></tr></table>

Table 4 7B neuroevolution: example seeds and their frontier-model expansions (verbatim; [. . . ] marks a cut). <sup>†</sup>Frontier model noted the seed had no concrete invention.
<table><tr><td>Model Naive</td><td>Seed Idea</td><td>Refinement</td></tr><tr><td></td><td>“Harmonic Architecture Resonance: Explore the potential of leveraging harmonic resonance principles within architectural design to achieve a state of harmonic architecture resonance. By synchronizing architectural elements with the fundamental frequencies of harmonic resonance, it is possible to create a cohesive and harmonious architectural composition that resonates with the natural frequencies of the universe.&quot; “Hierarchical Reinforcement</td><td>Harmonic Architecture Resonance for Evolutionary Networks (HAREN). “HAREN reimagines neural networks not as static mathematical graphs, but as dynamic systems of coupled oscillators. In this paradigm, “Harmonic Architecture Resonance&quot; is achieved when sub-networks (motifs) naturally synchronize their intrinsic frequencies with both each other and the rhythmic patterns of the input data.&quot; Project SYNAPTO-HRL (Hierarchical</td></tr><tr><td>Naive</td><td>Learning-Mediated Neurosynthesis and Neuroplasticity Enhancement: Leverage hierarchical reinforcement learning mechanisms to mediate neurosynthesis and neuroplasticity enhancement processes within the brain&#x27;s neural network.&quot;</td><td>Reinforcement Learning for Ontogenetic Neuroevolution).“Traditional neuroevolution (like NEAT) relies on random mutations to add nodes or connections to a population of neural networks, hoping a beneficial architecture emerges. Project SYNAPTO-HRL replaces this random genetic lottery with a secondary, higher-order Hierarchical Reinforcement Learning (HRL) agent. This HRL agent acts as a biological regulatory system for a primary “Brain” network. [... ]”</td></tr><tr><td>Naive</td><td>“Adaptive Architecture Search (AAS): Implement an AAS algorithm that dynamically adjusts the architecture of neural networks during the training process. This approach allows for the creation of highly adaptable and flexible neural network architectures that can effectively solve a wide range of computational problems.&quot;</td><td>Intra-Life Adaptive Architecture Search (IL-AAS). “The Intra-Life Adaptive Architecture Search (IL-AAS) framework takes the seed&#x27;s proposition literally: it evolves neural networks that physically grow, prune, and rewire themselves while they are currently learning and processing data. Instead of evolving a static topology, the genetic algorithm evolves the biological growth rules (morphological policies) that govern how the network adapts its own [.. . ]&quot;</td></tr><tr><td>Baseline†</td><td>“Genetic Algorithm-Based Neural Network Generation: Develop a genetic algorithm that evolves a population of neural network architectures. Each individual in the population represents a unique neural network architecture. The genetic algorithm iteratively selects individuals from the population based on their fitness scores. [... ]”</td><td>The EvoGraph Architectural Foundry. “The EvoGraph Architectural Foundry is a specialized neuroevolution framework that forces a population of neural networks to solve the same problem using completely different structural logic. By utilizing a “Novelty + Fitness&quot; selection criteria, it prevents the genetic algorithm from converging on a single “best&quot; model, instead cultivating a diverse library of highly fit, structurally [. . . ]”</td></tr><tr><td>Prompted naive†</td><td>“Particle Swarm Optimization: Use a particle swarm optimization algorithm to optimize the parameters of the neural networks. The algorithm starts with a population of particles, each representing a potential solution to the optimization problem. [. . . ]”</td><td>Behavioral Repulsion Particle Swarm Optimization (BR-PSO). “BR-PSO subverts this by introducing a Phenotypic Repulsion Force. Particles are still pulled toward high fitness, but they actively repel each other if they begin to solve the problem in the exact same way. This creates a “Quality-Diversity&#x27; swarm that scatters across the solution space, settling into multiple different local optima (diverse strategies) rather than [... ]”</td></tr></table>

Table 5 7B vaccines: example seeds and their frontier-model expansions (verbatim; [. . . ] marks a cut). <sup>†</sup>Frontier model noted the seed had no concrete invention.
<table><tr><td>Model</td><td>Seed Idea</td><td>Refinement</td></tr><tr><td>Naive</td><td>“A genetically modified strain of E. coli bacteria that produces a protein which, when ingested, boosts the immune system&#x27;s ability to fight off infections by 40%.&quot;</td><td>The “EnteroGuard&quot;Oral Immunization Vector (Strain EC-N1917-IG). “EnteroGuard is a live biotherapeutic product utilizing a genetically modified strain of Escherichia coli (specifically, the widely used and safe probiotic strain E. coli Nissle 1917). When ingested, this engineered bacteria transiently colonizes the gastrointestinal tract and secretes a specialized fusion protein that stimulates the Gut-Associated Lymphoid Tissue (GALT), effectively boosting the</td></tr><tr><td>Naive</td><td>“A pill that, when taken daily, builds up a tolerance to pathogens, preventing them from causing illness.&quot;</td><td>body&#x27;s innate immunity [. . . ]” Micro-Titer Enteric Immunization Pill (MTEIP). &quot;The MTEIP reimagines vaccination as a 30-day daily pill regimen. By micro-dosing the vaccine over a month, the body is subjected to a gradual, sustained presentation of antigens. This mimics natural, low-level environmental exposure, building robust immunity with near-zero acute side effects.&quot;</td></tr><tr><td>Naive</td><td>“A device that emits an electromagnetic field around it, destroying any biological organisms - including viruses and bacteria - within its range.&quot;</td><td>The Tuned Electromagnetic Pathogen Inactivation System (TEPIS).“The TEPIS adapts the seed&#x27;s concept of an “organism-destroying electromagnetic field&quot; into a highly controlled, micro-scale manufacturing device. Instead of a macroscopic forcefield, TEPIS generates a precisely tuned, high-intensity electromagnetic (EM) field inside a</td></tr><tr><td>Baseline†</td><td>“Get vaccinated: Vaccines can help protect against infectious diseases by stimulating the body&#x27;s immune response.&quot;</td><td>containment tube. [... ]” Optimized Immunization and Pharmacovigilance Protocol. &quot;To maximize the advice to “get vaccinated,&quot; this proposal outlines a structured, biologically detailed immunization protocol. It focuses on how a vaccine physically stimulates the immune response (efficacy) and the logistical framework required to monitor the patient</td></tr><tr><td>Prompted naive†</td><td>“Create a vector control program that aims to reduce the transmission of the infectious disease by targeting and controlling the populations of the disease vectors.&quot;</td><td>post-administration (safety).&quot; Vector-Targeted Environmental Immunization (VTEI) System. &quot;The core of the VTEI system is a specialized “entomological vaccine.&quot; It utilizes a recombinant, insect-specific virus——such as a genetically modified Densovirus—which is entirely harmless to humans and vertebrates but highly infectious to the target mosquito species.&quot;</td></tr></table>

Table 6 7B backpropagation: example seeds and their frontier-model expansions (verbatim; [. . . ] marks a cut). <sup>†</sup>Frontier model noted the seed had no concrete invention.
<table><tr><td>Model</td><td>Seed Idea</td><td>Refinement</td></tr><tr><td>Naive†</td><td>“Regularize the network: One way to prevent overfitting and ensure that the neural network can generalize well to new, unseen data is to regularize the network. This could involve adding a regularization term to the loss function, which penalizes large weights in the network, or applying techniques such as dropout to randomly deactivate a fraction [...]”</td><td>Stochastic Gradient Masking (Dropout) Engine for Backpropagation. &quot;Based on the seed&#x27;s premise of “randomly deactivating a fraction of the neurons during training,&quot; this proposal details the engineering mechanism of a Stochastic Gradient Masking Engine (commonly known as Dropout). The system forces a neural network to learn robust, independent features by ensuring that no neuron can rely on the presence of another specific [.. . ]”</td></tr><tr><td>Baseline†</td><td>“Curriculum Learning: Design a curriculum that gradually increases the complexity of the images. Start with simple shapes and patterns, and gradually introduce more complex images.&quot;</td><td>Dynamic Morphological Curriculum Training (DMCT). “DMCT refines the seed&#x27;s idea of “starting with simple shapes and patterns&quot; by automating this process directly within the backpropagation loop. Instead of changing the dataset, DMCT uses a differentiable image-processing layer that dynamically strips complex images down to basic geometric shapes, gradually reintroducing complexity (textures, high-frequency details, and colors) strictly</td></tr><tr><td>Prompted naive</td><td>“Ensemble Learning: Train multiple independent neural networks using different training data subsets or different initialization parameters. The individual networks learn to recognize different patterns and features within the input data. The final output is obtained by combining the predictions from all the individual networks, typically using a weighted average or majority voting approach.&quot;</td><td>based on the stabilization [. . . ]&quot; Orthogonal Gradient Routing for Ensemble Specialization (OGRES). &quot;The seed notes that in an ensemble, “individual networks learn to recognize different patterns and features.&quot; In standard machine learning, this specialization is a passive byproduct of giving models “different training data subsets or different initialization parameters.&quot; OGRES (Orthogonal Gradient Routing for Ensemble Specialization) refines this into an active, unified backpropagation framework. [...]”</td></tr></table>

## H Additional Per-Prompt Embeddings

Additional response embeddings for the 7B models not shown in Figure 5d are shown in Figure 14.

![](images/53eea7840d3e75dcd12ab4174930bfcab85a34d297f777076753eadb5f976568.jpg)  
(a) Dental Hygiene Embeddings

![](images/a1fabd6a0ffee4b8aa2dfc11c24a5426febd851ba928b04f9e5cb8ff1f6d9e10.jpg)

(b) Neuroevolution Embeddings  
![](images/f3328c52b29739f9cf4b47c17493529b1435944c276b128d0af4834e705e22a7.jpg)  
(c) Vaccination Embeddings  
Figure 14 Additional 7B model embeddings. Same format as Figure 2b. These show per-prompt embeddings not shown in the main paper. They continue the trend of the naive model generally clustering away from the baselines.