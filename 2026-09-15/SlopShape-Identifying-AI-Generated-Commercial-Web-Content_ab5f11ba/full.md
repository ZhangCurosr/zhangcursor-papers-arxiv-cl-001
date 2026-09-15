# SlopShape: Identifying AI-Generated Commercial Web Content

Jochen Madler

Sitefire

jochen@sitefire.ai

## Abstract

Word-level detectors identify unedited AIgenerated text almost perfectly, but the literature documents their brittleness under rewording, and a word-level score neither characterizes a text nor identifies which AI model wrote it. We ask whether AI-generated text can be identified one level deeper, from struc tural signatures: how information is presented, in what order, with what evidence, and in what voice. We replicate StoryScope (Russell et al., 2026a), which showed such patterns for AI-generated fiction, on commercial content: 2,250 pre-ChatGPT human blog posts from 268 company domains against 11,250 AI mirrors from five frontier models. A 214-feature instrument, applied by an LLM and validated in a human gold-annotation session (humanhuman kappa 0.928, human-model 0.946), detects AI posts from its 187 structural features alone at 98.0 macro-F1 on held-out companies, unchanged (98.1) when every AI post is reworded by its own model. The signal characterizes and attributes: AI posts share a tidy, self-announcing shape, 79.3% are attributed to the correct source against a 16.7% chance rate, and human posts occupy rare structural configurations. All effects replicate StoryScope’s, consistent in direction and larger in magnitude. We release pipeline, instrument, prompts, code, and aggregate artifacts.

## 1 Introduction

In commercial search engines like Google or Bing, text carries direct economic value: whether a web page ranks in the top results is directly tied to brand visibility and web traffic. With the rise of AI search engines like ChatGPT and Google AI Mode, web pages are no longer subject to a search engine’s algorithm alone, but are also assessed by an LLM. This additional step has changed the rules of Search Engine Optimization (SEO) and sparked a successor practice, Generative Engine Optimization (GEO): editing web page content can lift a page’s visibility in AI search answers by up to 40 percent, while traditional tactics like keyword stuffing yield little to no improvement (Aggarwal et al., 2024), and AI search engines cite systematically different pages than traditional rankings while reducing clicks to the sources they summarize (Chen et al., 2025; Khosravi and Yoganarasimhan, 2026). Section 2 examines the literature in more detail.

At the same time, LLM adoption has driven the marginal cost of producing content toward zero, and low-quality AI-generated text is now so common that it has its own term, “slop” (Shaib et al., 2025). Academic studies of professional content place the share of LLM-assisted text between 9 and 24 percent: about 9 percent of US newspaper articles and up to 24 percent of corporate press releases (Liang et al., 2025). Industry analyses report even higher numbers: Graphite (2025; 2026) estimates that roughly half of newly published English web articles are primarily AI-written, and Ahrefs (2025) finds some AI involvement in about three quarters of newly crawled pages. The share of AI-generated text is highest in the commercial web, where pages are most directly coupled to traffic and business incentives. We therefore study AI-generated text where it is most prevalent: commercial web pages, in particular company blog posts.

For identifying AI-generated content, word-level detection works well: on our corpus, a fine-tuned encoder distinguishes unedited AI-generated posts from human writing almost perfectly at near-zero cost (Section 5). Word-level detection has one weakness, however. Changing words is easy, many free tools automate it, and every word-level detector family has a documented failure mode under such adversarial rewording (Krishna et al., 2023; Weber-Wulff et al., 2023). A commercial ecosystem of “humanizer” tools has industrialized exactly this attack (Masrour et al., 2025). So we look one level deeper, for patterns that are much harder to rewrite. We test this directly: rewording every AI post in the test split leaves structural detection unchanged (Section 5.4).

Because word-level detection of reworded AIgenerated text is unreliable, policies of platforms like Google have adapted. Google’s policy on scaled content abuse targets outcomes rather than methods, sanctioning mass-produced pages “generated for the primary purpose of manipulating search rankings and not helping users” whether “automation, humans or a combination are involved” (Google, 2024). And instead of words, YouTube uses structural signals like account coordination, upload pacing, and templated narrative patterns (Mathur et al., 2026).

In this paper, we focus on these structural patterns the platforms observe, and ask whether AIgenerated commercial content has a structure, too. And if it does, what that structure looks like: which points AI-generated text makes, in what order, with what evidence, and in what voice - independent of the words being used.

This question has already been answered once, for fiction. StoryScope (Russell et al., 2026a) showed that AI-generated fiction can be told apart from human fiction by narrative structure alone (93.2 macro-F1 without stylistic signals), with humans occupying statistically rarer regions of narrative space. But fiction is not where AI adoption and economic incentives concentrate. We therefore carry their question, alongside the validated pipeline and methodology, to a domain where text is tied to revenue: commercial blog content. Our claims are focused on detection of single-pass AIgenerated blog posts from the five studied AI models, both as generated and after rewording (Section 9).

Figure 1a gives an overview of the study design, and Figure 1b previews the central result. The core methodology of the original paper is held fixed: the same five-stage pipeline (templates, cross-source comparison, feature discovery, deduplication, feature application), the same five AI models, the same classifier protocol and metrics. Our methodology is adapted where the domain demands it, and each adaptation is declared in a deviation register (Section 4.9). Our contributions are:

• The shape of AI-generated commercial writing, and a validated instrument that reveals it. AI-generated posts share a measurable structural signature, and ten features carry most of the signal (Section 6). The instrument is an 11-dimension commercial template schema discovered bottom-up from human posts, carrying 214 validated features (Section 4). Because the pipeline is LLM-run end to end, we audit it: repeatability runs, a reliability filter applied before any human/AI labels were seen, and a human gold-annotation session that exceeds the original’s agreement numbers (Section 7).

• Attribution and characterization. The shape identifies its author: 79.3 percent of posts are attributed to the correct one of the six sources, against a 16.7 percent chance rate. The shape is also shared: all five AI models crowd into the same common structural configurations, while human posts occupy disproportionately rare ones (Sections 5-6).

• Detection performance, via replication. Structural features alone detect AI-generated commercial blog posts at 98.0 macro-F1 on held-out data from companies never seen in training, consistent with the original study and larger in magnitude (Section 5). The signal is unchanged when every AI post is reworded by its own model (Section 5.4).

• A verifiable release. Our data acquisition pipeline, the full instrument, all prompts, code, and the aggregate artifacts behind every exhibit are public, with post-level data available to researchers on request.

## 2 Related Work

AI search and the economics of AI-generated content. AI search engines answer user prompts with LLM-generated text grounded on retrieved web pages. Aggarwal et al. (2024) define the term GEO and show that modifying web page content can lift visibility in AI answers by up to 40 percent while traditional keyword stuffing does not. AI search engines cite systematically different sources than classic rankings, with top-5 overlap of only 15- 33% (Chen et al., 2025), and getting cited correctly has itself become an optimization target (Tian et al., 2026). On the user side, AI-generated summaries reduce clicks on the pages they cite: exposure to Google’s AI Overviews reduced traffic to certain Wikipedia articles by about 15 percent (Khosravi and Yoganarasimhan, 2026). On the producer side,

![](images/55965eb6d58dd7b3a7b98ea34e630cf82b0282cd54fa4398ec3cd932dd4e6a7b.jpg)

(a)  
![](images/c2a734df623b7db830670fa13468cf2bfbbdb7a8bed8175ae9da61a245bfbac0.jpg)  
(b)  
Figure 1. (a) Study pipeline, from corpus construction through template extraction, feature discovery, and scoring to classification, the rewording test, and validation. (b) The central result previewed: structural rarity percentile by source. Human posts concentrate in the rarest structural configurations (mean 0.84 vs pooled AI 0.44, Cohen’s d = 1.83), while the five AI models crowd the common ones. Section 6 develops this result.

AI search measurably shifts what content communities publish (Zhang et al., 2026).

Adoption measurement. Academic studies already find a sizeable share of AI-generated text in professional writing: about 9 percent of new US newspaper articles (Russell et al., 2026b), up to 24 percent of corporate press releases (Liang et al., 2025), and more than 5 percent of new English Wikipedia articles (Brooks et al., 2024). At web scale, a large share of multi-language page translations is already mass machine translation, skewed toward low-quality SEO content (Thompson et al., 2024). Industry measurements report even higher shares (Section 1).

AI-text detection. Word-level detectors are the first choice for AI-text detection, and on unedited text they are close to perfect. They span fine-tuned encoders (ModernBERT; Warner et al., 2024), stylometric classifiers, TF-IDF baselines, and zeroshot likelihood methods (DetectGPT, Mitchell et al., 2023; Binoculars, Hans et al., 2024). Their weakness is rewording, which is easy and widely automated. All four families have documented failure modes under adversarial rewording (Sadasivan et al., 2023). Likelihood-based zero-shot detectors collapse under paraphrasing: DetectGPT falls from 70.3 percent to 4.6 percent detection at 1 percent false positives (Krishna et al., 2023). Trained classifiers degrade on unseen models, domains, and edits, with accuracy on AI-paraphrased text as low as 26 percent (Weber-Wulff et al., 2023). Lexical-tell estimators are defeated by avoiding the tell words (Kobak et al., 2025). Provider-side watermarks like the one Anthropic introduced in August 2026 survive light editing but not a complete rewording, by Anthropic’s own account (Anthropic, 2026). The rewording attacks are heterogeneous and hard to anticipate (Dugan et al., 2024), and there is a commercial tool ecosystem that exploits them (Masrour et al., 2025; Perkins et al., 2024). The most rewriting-robust result comes from the study this paper replicates: StoryScope’s narrative features lost only 1.6 F1 under span-level style rewriting, using the editing protocol of Chakrabarty et al. (2025). We reproduce their word-level baseline suite on our data (Section 5) and run their rewording test on commercial text (Section 5.4).

Structural analysis of AI writing, with LLMs as annotators. StoryScope (Russell et al., 2026a) built on the NarraBench taxonomy of narrative dimensions (Hamilton et al., 2025) to measure discourse-level differences between human and AI fiction, and found structural detectability, modelspecific patterns, and a rarity gap. LLM-prose idiosyncrasies have also been formalized from the editing side (the LAMP taxonomy; Chakrabarty et al., 2025) and the judging side (Shaib et al., 2025). This paper is a domain-transfer replication of StoryScope, and every deviation from the original is disclosed in a register (Section 4.9). Our pipeline uses LLMs for screening, template extraction, feature discovery, and scoring. We follow the original paper’s gold-annotation protocol and extend it with a reliability filter that never uses the human/AI labels (Section 4.5) and a human goldannotation session (Section 7).

## 3 Data

Our human writing corpus comprises 2,250 human commercial blog posts from 268 company domains, captured in pre-ChatGPT Wayback Machine snapshots dated between 2008 and 2022. The posts come from company websites that sell to other businesses (B2B) and for which organic search is a plausible acquisition channel. Table 1 describes the collection pipeline: we assembled a sampling frame from four public company lists, screened the companies for fit, kept domains with enough articles archived before ChatGPT’s release, spotchecked their genre, applied industry quotas, and fetched the qualifying articles through the content filters (600-2,500 words, English, informational genre, near-duplicate removal). Of the 306 selected domains, 264 yielded usable posts (40 had no fetchable qualifying articles, 2 were emptied by the content filters), and 4 replacement domains drawn from the same qualification pool during the fetch gave the 268 corpus domains (see Appendix A). Following the original study’s terminology, we define a “prompt” as one human blog post together with its brief and the five AI-generated mirror posts derived from it. A seeded, stratified set of 100 prompts, spanning 82 of the 268 domains, is set aside for feature discovery (Section 4.3) and excluded from all classifier splits. We call its 600 posts the featurediscovery set. The remaining 2,150 prompts form the classification corpus, split 198/32/38 by company domain into train/val/test (Section 4.7).

The corpus spans seven industry verticals from software/SaaS (26.5%) to edtech (0.6%), with the full composition by vertical, source frame, and snapshot year in Appendix A (Table A1). Because we collected as much data as possible from just before ChatGPT’s release, the corpus skews recent: 75.5% of post snapshots are from 2020-2022. We address this with a publication-year check (Appendix G) and disclose it as a limitation (Section 9).

Following the original study’s reverseengineering design, a content brief is inferred from each human post, with publisher identity and meta labels anonymized so they cannot influence the brief. Each brief is then given to the original’s five AI models: gpt-5.4, claude-sonnet-4.6, gemini-3-flash, deepseek-v3.2, and kimi-k2.5, yielding 11,250 AI-generated mirror blog posts. Together, the human posts and their mirrors form a paired corpus: for every human post there are five AI-generated versions written from the same brief. AI mirrors turn out to be slightly longer than their human counterparts (per-model means of 1,064 to 1,541 words, length statistics in Appendix A). To rule out effects driven by post length, we ran a length check (Appendix G) as well as a length-only classifier: given only word count, it cannot separate the classes (Section 5).

In all LLM-facing prompts, the web page identity is anonymized. Schema discovery, the stage that derives the 11 description dimensions (Section 4.1), saw only human posts. Feature discovery, the stage that derived the 457 candidate features within those 11 dimensions, saw posts from all six sources without knowing which was which. Nevertheless, a human post could be recognized for a trivial reason: it might be part of an AI model’s training data. To rule this out, we ran a memorization check using the 13-gram overlap rule of Brown et al. (2020): if a human post and its mirror share even one identical 13-word sequence, the pair is flagged as potentially contaminated. Only 0.19% of pairs were flagged (against 0.0% for a shuffledhuman control), no pair was near-verbatim, and excluding all flagged prompts leaves the headline unchanged in substance (97.9). As a final check, we scanned all posts for mentions of entities that did not yet exist at the post’s claimed publication date. Only 0.08% of posts contain such a mention, and excluding them leaves the headline numbers unchanged.

<table><tr><td>Step</td><td>Rule</td><td>Count</td></tr><tr><td>1. Assemble the sampling frame</td><td>Sampling frame assembled from four archived public company lists: Inc5000 9,979; YC directory 3,448; FT1000 1,129; G2 519 (with an anti-persona prefilter)</td><td>15,075 domains</td></tr><tr><td>2. Company-fit screen</td><td>2 LLM judges assess the company fit from firmographics and homepage metadata</td><td>94.3% keep</td></tr><tr><td>3. Archive-volume check</td><td>At least 25 article URLs archived before 2022-11-30 in the Wayback Machine&#x27;s snapshot index</td><td>698 qualified</td></tr><tr><td>4. Genre spot-check</td><td>5 posts/domain checked for English language, informational type, and web page length</td><td>307 keep-eligible</td></tr><tr><td>5. Industry quotas</td><td>Company industry quotas applied (software cap 40%)</td><td>306 domains</td></tr><tr><td>6. Fetch and filter</td><td>Content filters: 600-2,500 words, English, informational genre, near-duplicate removal</td><td>2,250 posts from 268 domains</td></tr></table>

Table 1 - Data collection pipeline

## 4 Measuring Structure

The measurement pipeline follows the original’s five stages, preceded by one new schema-discovery step. Figure 1a gives the overview.

## 4.1 Commercial template schema

Since the original’s template schema is fictionshaped (plot, characters, emotion trajectories), we derived a similar schema for commercial content bottom-up: three independent runs of the same LLM (gpt-5.6-terra) each examined 18 stratified human posts and converged on near-identical dimension systems. The consolidated result is a new 11-dimension schema: purpose, audience, structure and flow, explanation, evidence, voices, actionability, commercial integration, timeliness, page format, and writing style. We kept a dedicated Writing-Style dimension so that style is measured on its own and can be excluded. Every structural result in this paper is computed with these writing style features stripped out, which is what lets us attribute the detection signal to structure rather than wording. In Appendix B, we map the NarraBench fiction dimensions to our commercial ones.

## 4.2 Template extraction

For each of the 13,500 posts - the 2,250 human posts and their 11,250 AI mirrors, six posts per prompt - we extracted one JSON template that describes the post along all 11 dimensions of the schema.

## 4.3 Comparison, feature discovery, quality gate, deduplication

Using the extracted templates, we ran the original study’s cross-source comparison stage over the feature-discovery set: the 100 prompts (600 posts) set aside for this purpose and excluded from all classifier training and testing, so that features are never discovered on the same posts they are later evaluated on. The feature-discovery set shares some company domains with the test split, and a dedicated check covers this overlap (Appendix G). For each of the 100 prompts, gpt-5.6-terra received all six templates - the human post’s and its five AI mirrors’ - side by side and noted where the sources diverge. Three independent LLM discovery runs, each with one specialized prompt per dimension, then turned these divergence notes into 457 features, versus the original’s 408. We then applied a quality gate to score each feature on whether it can be answered from the text alone, with locatable evidence, in self-explanatory wording, and without compound constructs. The gate rejected 38.3% of candidates (see Appendix C), leaving us with 282 features. Finally, we removed duplicate features by clustering their embeddings (F2LLM-4B, singlelinkage clustering at cosine 0.85) and keeping only one feature per cluster, which left 266 features. In this embedding step, only 5.7% of our features were merged versus the original’s 25.5%, possibly because of our quality gate. We include a sweep on the deduplication cutoff threshold that shows the results do not depend on this difference (Appendix G).

## 4.4 Feature application

To turn features into measurements, every blog post is scored against every feature. We used gemini-3.6-flash, one dimension per call, with the model forced to pick exactly one answer from each feature’s predefined options. Before committing to the full corpus, we ran two checks on a small sample. First, scoring one dimension at a time answered 99.97% of items, versus 99.75% for scoring the whole post in a single call. Small as the difference is, we kept the per-dimension mode (Appendix C). Second, we scored 60 of the posts five times, and agreement across the runs reached a Krippendorff alpha of 0.891 against our 0.8 bar (Section 7). We then scored all 13,500 posts against the 266 features. This produced 148,500 answers (13,500 posts x 11 dimensions).

## 4.5 Feature reliability filter

An LLM judge can produce unreliable answers: features whose answer values barely vary, that drift outside the predefined options, or that randomly change from run to run. To remove such features from further analysis, we ran a reliability filter that judges a feature by its result statistics. It excluded 52 features: 11 were degenerate (at least 98% of posts received the same answer), 3 produced answers outside their predefined options too often (above 2%), and 38 were unstable across the five repeat scoring runs. The final instrument therefore has 214 features (versus the original’s 304): 187 structural ones (the original’s “narrative-strict” analogue) and 27 style. Appendix C (Table C1) gives the composition by dimension and answer type.

## 4.6 Feature space, variants, encoding

Our encoding follows the original paper’s methodology: one-hot for nominal and binary types, multihot for multi-select, integer position encoding for ordinal and scale, NaN for missing (native to XG-Boost; Chen and Guestrin, 2016). The result is a matrix of 12,900 posts x 868 columns: each of the 2,150 classification prompts contributes six posts, and 600 posts from the initial feature-discovery set are excluded. As in the original, we train classifiers on subsets of the instrument: 187 structural features only, style features only (27), all features (214), the ten core features (Section 6), and the ten core features plus model-specific features (33 in total, Section 6).

## 4.7 Experimental protocol

Before training any classifier, we split the corpus by company domain: 1,566 / 294 / 290 prompts (198/32/38 domains), so no domain appears in more than one split. This is stricter than the original’s random prompt split (deviation D6) and prevents a company’s tone of voice from leaking across splits. Per the original’s protocol, we gridsearched hyperparameters on the validation split and trained the final models on training and validation data combined (see Appendix D). Our results are, unless stated otherwise, measured only on the held-out test split: macro-F1 and AUPRC for the human-vs-AI task (Section 5.1 explains this choice), and macro-F1 and accuracy for source identification, the task of predicting whether a post was written by a human or, if not, which of the five AI models wrote it. Confidence intervals come from 10,000 bootstrap resamples. Because posts from the same company resemble each other, the primary intervals resample whole company domains rather than individual prompts.

Hypotheses. We test three hypotheses. The first: style features outperform structural features, the opposite of the original study’s finding. The rationale: commercial writing is persuasion in a brand voice, so style could plausibly carry more of the signal than it does in fiction. Our data rejects this hypothesis decisively (Section 5.3). The second: structural detection is robust against rewording, because the features capture what a post says rather than how it formulates it. It holds (Section 5.4). The third: the gold-annotation session must reach human-human and human-model agreement of at least kappa 0.60. Both bars were cleared at 0.928 and 0.946 (Section 7).

## 4.8 Rewording test

To test the second hypothesis, every AI post in the test split is rewritten by the same AI model that generated it. The rewriting follows the editing protocol of Chakrabarty et al. (2025): the model edits its own post span by span, targeting the seven categories of AI-writing artifacts their study identified with professional editors, while keeping every claim, fact, and link. We ensured preservation of content after rewording by running a judge on all 1,450 original-rewritten pairs. In the final pairs, 98.2% of blog posts preserve all claims and the rest differ in single details. The rewriting attack is substantial: on average, 73% of a post’s 13-word sequences no longer appear verbatim, and 15.6% of the measured structural feature answers change.

## 4.9 Declared deviations

Table 2 lists the deviations from the original’s method that most affect how the results should be read. The full register with all 16 deviations and their defenses is in Appendix E.

## 5 Detection Results

## 5.1 Binary detection

We report macro-F1, an accuracy measure balanced across two classes: it averages the detection quality on human-written and on AI-generated posts equally. Naive accuracy would be misleading, because five of every six posts are AI-generated: a detector that answers “AI” for everything already reaches 83.3% accuracy without catching a single human post.

The first question is how well structural features alone can distinguish AI-generated from humanwritten blog posts, and how close they come to the word-level baselines. Table 3 reports classifiers trained on each subset of our instrument as well as the word-level baselines from the original study, all evaluated on the same held-out test data.

Result: We replicate the original study’s headline result at larger magnitude. Structural features alone distinguish AI-generated from humanwritten commercial blog posts at 98.0 macro-F1. The ordering of the feature variants replicates, too: all features (98.1) above structural only (98.0) above style only (88.1), against the original’s 96.0 $> 9 3 . 2 > 8 5 . 8$

Word-level detection: the framing of Section 1 holds on this corpus. Unedited single-pass generation is near-perfectly detectable at the word level: the stylometric baseline and ModernBERT both reach a perfect score, mirroring the original’s 99.9 and 99.8. Part of the stylometric ceiling comes from formatting rather than wording. Some human posts are interviews, transcripts, or roundups, formats the AI mirrors almost never produce. Restricted to the 86.5% of human posts that are regular self-contained pages, the structural headline remains 97.5. The zero-shot substitute (a small Qwen2.5-0.5B pair in place of the original’s Falcon-7B Binoculars pair) lands at 78.2 where the original’s true Binoculars scored 55.9. Moreover, blog post word count by itself cannot separate the classes, consistent with the length check (Appendix G).

As in the original, all feature-based models and the stylometric and TF-IDF baselines are trained on the combined training and validation data. The ModernBERT baseline is an exception: per the original study, we fine-tune it on the training split alone. Two more baselines, a zero-shot detector substitute and a length-only control, are reported in Appendix F, and all values are percentages reported to one decimal. AUPRC measures ranking separation rather than thresholded decisions and is rounded to three decimals (the structural model’s stored value is 0.9998), so the structural variants’ remaining test errors are threshold errors, not ranking errors. The results are not sensitive to hyperparameter tuning: refitting with the original study’s published hyperparameters yields only a 0.2 points difference compared to our tuned configuration on both tasks (Appendix D).

## 5.2 Distance to the word-level ceiling

The best word-level detectors on this corpus are perfect, but the structural model comes within two percentage points: 98.0 macro-F1 against their 100.0 (the original: 93.2 against near-perfect word-level baselines). These word-level detector ceilings are measured on unedited text, and there is extensive literature on the brittleness of word-level detectors against rewording attacks (Section 2). In contrast, our structural feature instrument’s accuracy remains unchanged under rewording attacks (Section 5.4).

## 5.3 The style-vs-structure direction

The first hypothesis (Section 4.7) predicted that style features would beat structural features. The data reverses it: the structural model beats the style model by 9.9 macro-F1 points (95% CI [6.0, 14.2] on the primary company-cluster basis, and [7.6, 12.4] at the prompt level).

Of the 1,740 test posts, the structural model makes 19 errors and the style model 101, and only 4 posts are misclassified by both. If structural features merely re-encoded surface style, the two models should fail on the same posts. Instead, their failure sets are nearly disjoint, which is evidence that the style features and structural features measure two different signals (see Section 8).

## 5.4 Detection survives rewording

The second hypothesis holds. On the test split with every AI post replaced by its reworded version, the structural classifier scores 98.1 macro-F1, against 98.0 on the original posts (95% CI on the reworded split 97.0-99.1). Of 1,450 reworded AI posts, only

<table><tr><td>#</td><td>Deviation from the original</td><td>Defense</td></tr><tr><td>D1</td><td>Domain: B2B blog posts</td><td>The research question</td></tr><tr><td>D6</td><td>Domain-disjoint splits, single-unblinding holdout</td><td>Stricter than their random prompt split</td></tr><tr><td>D7</td><td>Rarity metric re-implemented from text</td><td>Verified on their data</td></tr><tr><td>D10</td><td>Corpus ~1/4.5 of theirs, feature-discovery set 4.4% vs ~1%</td><td>Budget, power note in Section 9</td></tr><tr><td>D11</td><td>Commercial-native schema discovered from human posts</td><td>Fiction lens misfits the domain</td></tr><tr><td>D13</td><td>Evaluation-stage models upgraded within lineage</td><td>The five AI models unchanged</td></tr><tr><td>D14</td><td>Rewording test: symmetric self-rewrite by all five models</td><td>Removes the original&#x27;s single-model asymmetry</td></tr></table>

Table 2 - Deviation register (condensed to the most consequential rows).
<table><tr><td>Variant / baseline</td><td>n feat</td><td>Test macro-F1</td><td>AUPRC</td><td>Original</td></tr><tr><td>Structural (headline)</td><td>187</td><td>98.0 (cluster CI 96.7-99.2, prompt CI 97.1-98.9)</td><td>1.000</td><td>93.2 / .959</td></tr><tr><td>Style-only</td><td>27</td><td>88.1</td><td>.994</td><td>85.8 / .867</td></tr><tr><td>All features (combined)</td><td>214</td><td>98.1</td><td>1.000</td><td>96.0 / .982</td></tr><tr><td>Core-only</td><td>10</td><td>93.5</td><td>.996</td><td>30 feat: 84.8 / .828</td></tr><tr><td>Core + model-specific</td><td>33</td><td>95.9</td><td>.999</td><td>101 feat: 91.1 / .934</td></tr><tr><td>ModernBERT-base fine-tune</td><td></td><td>100.0</td><td>1.000</td><td>99.9 / 1.00</td></tr><tr><td>Stylometric + XGB (144-dim, their spec)</td><td>144</td><td>100.0</td><td>1.000</td><td>99.8 / .999</td></tr><tr><td>TF-IDF (1-2g, their 5,000-feature spec) + XGB</td><td>5,000</td><td>99.3</td><td>1.000</td><td>99.7 / .999</td></tr></table>

Table 3 - Binary human-vs-AI detection on the held-out test split (macro-F1, original’s Table 2 analogue).

<table><tr><td>Variant</td><td>Original</td><td>Reworded</td></tr><tr><td>Structural (187)</td><td>98.0</td><td>98.1</td></tr><tr><td>Style-only (27)</td><td>88.1</td><td>87.1</td></tr><tr><td>Ail features (214)</td><td>98.1</td><td>97.9</td></tr></table>

Table 4 - Detection on the original versus reworded test split (macro-F1).

5 are misclassified as human-written. The result is uniform across the five AI models (97.6-98.1). Style features, which read how a post is worded, lose ground as expected under a wording attack (88.1 to 87.1), and the combined set is essentially unchanged (98.1 to 97.9). The changed feature answers do not add up to an escape: they are scattered across posts and features rather than directed toward the structural profile typical of human posts, and a classifier that reads the whole 187-feature signature is barely affected. The original study observed the same for its narrative features in fiction, with a 1.6-point drop on a single-model rewording arm. Ours holds across all five models. The rewriting protocol and verification gates are in Section 4.8 and Appendix J.

<table><tr><td>Class</td><td>Ours</td><td>Original (without style)</td></tr><tr><td>human</td><td>.966</td><td>.89</td></tr><tr><td>gpt</td><td>.855</td><td>.73</td></tr><tr><td>claude</td><td>.830</td><td>.77</td></tr><tr><td>kimi</td><td>.747</td><td>.55</td></tr><tr><td>gemini</td><td>.698</td><td>.60</td></tr><tr><td>deepseek</td><td>.654</td><td>.57</td></tr></table>

Table 5 - Per-class F1 for the six-source model (original’s Table 11 analogue).

## 5.5 Identifying which AI model wrote a post

We also trained a model that identifies each post’s author among the six sources: the human author or one of the five AI models. On the test split, it assigns 79.3% of posts to the correct source (macro-F1 79.2, against a 16.7% chance rate), versus the original’s 68.4%. Table 5 shows the perclass scores.

Human-written posts are identified almost perfectly. The remaining errors are AI posts attributed to the wrong AI model (Figure 2). Posts by DeepSeek V3.2 are the hardest to attribute, which is consistent with DeepSeek’s position in structural feature rarity space (Section 6).

![](images/62c57bc5ff5cad1665aa04e8e16b9e16a397268c62a3819949d8741490476925.jpg)  
Figure 2. Row-normalized test confusion of the sixsource model (accuracy 79.3% vs the 16.7% chance rate, macro-F1 79.2). Humans are near-perfectly separated. The residual confusion is between AI models, hardest for DeepSeek V3.2.

## 6 The Shape of AI Writing

Figure 3 shows the twenty most important features, ranked by mean absolute SHAP contribution. SHAP (SHapley Additive exPlanations; Lundberg and Lee, 2017) assigns each feature its share of every prediction the model makes. The most important features describe argument-structure stages, purpose and outcome framing, how the reader is addressed, and how sources are disclosed.

Using the original’s core-selection procedure, we then distilled this ranking down to a core: the small set of feature values that carry the signal most reliably. A value counts as core if it ranks in the top quartile of SHAP importance, stays important when the model is refit 50 times on resampled data, carries clearly more weight than it would with randomly shuffled labels, and separates human from AI answers by a wide margin that is consistent across all five AI models. This yields 10 core values on 10 distinct features (the original: 33 values on 30 features). These ten features account for 27.6% of the full model’s total feature attribution, and a classifier restricted to them alone still tests at 93.5 macro-F1 (the original: 30 features at 84.8). The small-core phenomenon replicates, with a smaller core and a higher accuracy, potentially because of the quality gate in our feature discovery and because commercial blog posts are more homogeneous than fiction.

Each core value points toward one author class, and together the directions form a consistent pattern. We interpret only concrete, directly observable features, within what the validity results of Section 7 support. The values most commonly produced by AI models assemble into what we call the tidy, self-announcing blog post: it promises the payoff already in the title, states its thesis and announces its structure before the first section, speaks in an editorial-explainer voice, and closes with a stage that summarizes or restates the thesis. In a typical AI post, the payoff is promised in the title: “How to Cut Onboarding Time in Half.” The thesis is stated and the flow announced before the first section: “In this post, we’ll cover why onboarding stalls, three fixes that work, and how to measure the difference.” And the close restates the thesis: “In short, structured onboarding saves time.” The human-leaning values describe posts without those signposts: no announced section flow, no escalation of stakes, and no closing section that restates the thesis. The length-matched check in Section 7 shows the headline result does not depend on post length.

Rarity. Following the original study and the tradition of Torrance (1966), we measure statistical rarity as a proxy for originality. Rarity is a per-post statistic: for each post, we compute the mean Euclidean distance to its 25 nearest neighbors in the z-scored structural feature space, and convert it to a percentile against the pooled train+val distribution as reference. A post with a common structure sits close to many neighbors and scores low. A post whose structure few others share sits far from its neighbors and scores high. We compute these statistics over all 12,900 posts, and on the original’s test-only basis the picture is the same (rarest 1%: 40 human vs 1 AI, full tail table in Appendix F). Human posts occupy the rarest regions of this space and AI posts the densest (Table 7 and Figure 1b): the mean human rarity percentile is 0.838 against 0.435 for AI, a gap of Cohen’s d = 1.83 (the original: 0.71 vs 0.49, d = 0.83). The contrast is even sharper in the tail. Among the posts whose rarity exceeds the 99th percentile of the reference distribution, 149 are human and 4 are AI, where the original study’s rarest 1% split roughly evenly (42 vs 41). The gap is equally visible in ranking terms: rarity alone separates the classes at AUC 0.901 (original 0.73), and the human post is the rarest of its six-post prompt group in 85.5% of prompts, against a 16.7% chance rate (original 57.8%). Per-model mean rarity orders the AI models deepseek 0.547 > claude 0.483 > gemini 0.461 > kimi 0.356 > gpt 0.329. DeepSeek sits closest to humans, matching its status as the hardest source to attribute, and the length-matched check (Appendix G) shows this proximity is structural rather than length-driven: the per-model ordering is preserved on the length-matched subset.

Top-20 structural features separating human from AI posts  
![](images/e16a3f88ef72378a5ea1616188c134fc9c6f166286a9c0cae6a67de447c99798.jpg)  
Figure 3. Top-20 features of the structural classifier by mean absolute SHAP contribution across bootstrap refits, labeled with their plain-language instrument names (feature IDs in parentheses, full question wording in the released instrument). Dark bars mark the core features (Section 6) that appear among the top 20.

The rarity gap also has a geometric reading: human posts are spread out widely in structural feature space while AI posts cluster together (Appendix F). From the six-source model’s per-class SHAP attributions, we selected 30 model-specific features - features whose importance concentrates on one source (human 17, gpt 6, deepseek 2, gemini 2, kimi 2, claude 1; the original: 75 features). A classifier on the ten core features plus these 30 model-specific features (33 distinct features) tests at 95.9 macro-F1, and the per-source feature lists are in Appendix F (Table F3).

## 7 Robustness and Instrument Validity

We ran eight robustness checks, every one under the original’s protocol (final models retrained on train+val). The rewording test, the ninth robustness result, attacks the input text rather than the pipeline and is reported with the detection results (Section 5.4). Table 8 summarizes them one line each, and the full table with complete results and caveats is Appendix G.

Two checks target the design confounds declared in Section 3. On a test subsample in which human and AI posts are matched on length (1,545 posts), detection is unchanged at 98.1 versus 98.0 unmatched (the original: 93.2 to 93.2), the rarity gap survives the matching (d = 1.87), and word count is nearly uncorrelated with rarity. Structural features also do not secretly encode when a post was written: predicting a human post’s publication period from them performs at chance (macro-F1 49.0), and removing the most year-associated features leaves the headline flat (97.8-98.0).

The remaining checks close the other declared exposures. Excluding the 0.19% of posts flagged by the memorization check leaves the headline at 97.9, and excluding the 11 company domains that the feature-discovery set shares with the test split leaves it at 98.2. Varying the deduplication cutoff between 0.70 and 0.95 changes the feature count (200 to 282) but not the conclusions. Discovering features directly from raw text instead of from templates yields as many candidates but substantially different ones (about 23% overlap), so the discovery pathway shapes which features are found - and the template pathway is the one validated end to end. Detection strength does not vary significantly across industry verticals (p = 0.109, per-vertical macro-F1 between 96.5 and 100), consistent with the original’s finding of no topic effect. Finally, classifier performance saturates by roughly 75% of the training data, so corpus size is not what limits the headline (Appendix G).

<table><tr><td>Feature</td><td>Value</td><td>Leans</td><td>Human-AI gap</td></tr><tr><td>Conclusion Behavior (STR_STG_008)</td><td>restated thesis or reframe</td><td>AI</td><td>0.650</td></tr><tr><td>Payoff First-Promise Location (PUR_OUT_003)</td><td>title</td><td>AI</td><td>0.624</td></tr><tr><td>Functional Stage Inventory (STR_STG_001)</td><td>summary or synthesis</td><td>AI</td><td>0.609</td></tr><tr><td>External Participation Pathway (VOC_PRT_005)</td><td>absent</td><td>AI</td><td>0.592</td></tr><tr><td>Legacy-versus-Modern Contrast (STR_FLW_005)</td><td>no</td><td>human</td><td>0.497</td></tr><tr><td>Thesis Before First Unit (STR_FLW_006)</td><td>no</td><td>human</td><td>0.419</td></tr><tr><td>Stakes Escalation (AUD_STK_005)</td><td>absent</td><td>human</td><td>0.362</td></tr><tr><td>Primary Voice Identity (VOC_VOX_001)</td><td>editorial explainer</td><td>AI</td><td>0.308</td></tr><tr><td>Article Length Class (PAG_FUR_011)</td><td>short, under 800 words</td><td>human</td><td>0.274</td></tr><tr><td>Problem Placement (AUD_PRB_002)</td><td>(ordinal position)</td><td>human</td><td>0.222</td></tr></table>

Table 6 - All core values, with the class each one points to (original’s Table 15 analogue).

<table><tr><td>Statistic</td><td>Ours</td><td>Original</td></tr><tr><td>Human mean rarity percentile</td><td>0.838</td><td>0.71</td></tr><tr><td>AI mean rarity percentile</td><td>0.435</td><td>0.49</td></tr><tr><td>Cohen&#x27;s d</td><td>1.83</td><td>0.83</td></tr><tr><td>Rarity AUC</td><td>0.901</td><td>0.73</td></tr><tr><td>Human rarest-of-prompt (chance 16.7%)</td><td>85.5%</td><td>57.8%</td></tr><tr><td>Rarest decile share (human</td><td>47.7% 1</td><td>24.7% 1</td></tr><tr><td>/ AI)</td><td>2.8%</td><td>7.1%</td></tr></table>

Table 7 - Rarity statistics (all 12,900 posts, percentile reference train+val).

Repeatability. We scored the same 60 posts five independent times with the scoring model. Agreement across the five runs is Krippendorff alpha 0.891 (Krippendorff, 2004; our 0.8 bar; original 0.90), mean pairwise Cohen kappa 0.890 (Cohen, 1960; original 0.89), and pairwise exact agreement 0.893.

Human gold validation. To validate that the LLM scorer measures what a careful human reader would measure, two annotators independently answered 20 features on 12 posts (240 items each), scored in the encoded representation. We fixed the acceptance thresholds before annotation began: human-human and human-model kappa of at least

0.60. Table 9 reports the result next to the original’s, with each kappa computed as Cohen’s kappa on the pooled item-level contingency over all scored items. Our human-human kappa is 0.928 against their 0.739, and our mean human-model kappa is 0.946 against their 0.839. The LLM scorer agrees with the human annotators more closely than the original’s did, and more closely than the original’s annotators agreed with each other. Each annotator’s individual agreement with the model clears the 0.60 bar as well. In a separate audit, the annotators reviewed the model’s calls on which features count as style rather than structure, and endorsed that boundary (all kappas above the 0.75 bar, detail in Appendix H).

Disclosures. The reliability filter’s 52 exclusions are itemized in Section 4.5, and the answer distributions, drift rates, and per-feature stability records are published in the release. Both annotators are company-affiliated (Section 9). An initial gold-session scoring pass was found to have run on placeholder sheet data and was redone on the final annotations. The reported numbers are from the corrected scoring, and both passes are in the release.

## 8 Discussion and Conclusion

We asked whether AI-generated text has a shape: a structural signature in how information is presented, in what order, with what evidence, and in what voice, independent of the words being used. Within the studied scope, the answer is yes. Commercial blog posts have no plot, no characters, and no narrative arc, yet the structural signature transfers. Detection performance shows the shape is a real signal and not an artifact of our instrument: structural features alone reach 98.0 macro-F1, and every phenomenon of the original study replicates in the new domain, consistent in direction and

<table><tr><td>#</td><td>Check</td><td>One-number result</td></tr><tr><td>1</td><td>Length confound</td><td>Headline 98.1 on a length-matched test subsample (vs 98.0 unmatched)</td></tr><tr><td>2</td><td>Publication-year check</td><td>Predicting a human post&#x27;s publication period from structure is at chance (macro-F1 49.0)</td></tr><tr><td>3</td><td>Template-vs-direct discovery</td><td>Same candidate yield, only ~23% semantic feature overlap</td></tr><tr><td>4</td><td>Deduplication cutoff sweep</td><td>200-282 features across cutoffs 0.70-0.95, sensitivity low</td></tr><tr><td>5</td><td>Feature-discovery-set domain overlap</td><td>Headline 98.2 excluding the 11 overlapping domains</td></tr><tr><td>6</td><td>Memorization</td><td>0.19% of pairs flagged, filtered headline 97.9</td></tr><tr><td>7</td><td>Vertical heterogeneity</td><td>No significant variation across verticals (Kruskal-Wallis p = 0.109)</td></tr><tr><td>8</td><td>Learning curve</td><td>Classifier performance saturates by ~75% of training data (98.0)</td></tr></table>

Table 8 - Robustness check summary (full versions in Appendix G).

<table><tr><td>Comparison</td><td>Agreement</td><td>Kappa</td><td>Original</td></tr><tr><td>Human-human</td><td>93.22%</td><td>0.928</td><td>76.85% 1 0.739</td></tr><tr><td>Annotator A vs</td><td>94.49%</td><td>0.9414</td><td>91.67% /</td></tr><tr><td>model Annotator B vs</td><td>95.34%</td><td>0.9502</td><td>0.9056 79.86% 1</td></tr><tr><td>model Mean</td><td></td><td></td><td>0.7724</td></tr><tr><td>human-model</td><td></td><td>0.946</td><td>0.839</td></tr></table>

Table 9 - Human gold validation (original’s Table 7, encoded basis, kappas computed as Cohen’s kappa on the pooled item-level contingency).

larger in magnitude.

The core feature values in Table 6 spell out a shape that we call the tidy, self-announcing blog post. AI-generated blog posts promise the payoff in the title, state their thesis before the first section, announce its flow, close with a summary, and speak in an editorial-explainer voice. The shape also identifies the author: our classifier assigns 79.3% of posts to the correct one of the six sources, far above the 16.7% chance rate (Section 5.5). And the distinct shape is shared across AI models: human posts sit in the rare regions of structural space, while the five AI models concentrate in common ones (Section 6). Mathur et al. (2026) observed at platform scale that mass-produced generative content can be seen as unique variations of functionally identical material. At the level of company blog posts, our measurement confirms this.

The shape also goes deeper than the world level. Rewording attacks replace most of the surface phrasing, but the structural feature detection accuracy remains unchanged (Section 5.4). The erroroverlap result points the same way: the structural and style models fail on almost entirely different posts, so the two feature sets carry different signals.

For search engine platforms, the practical reading is that structural features are a candidate signal for detecting AI-generated content that is robust against word-level attacks. Google’s policy on scaled content abuse already targets outcomes rather than methods (Section 1), and our structural instrument fits this reasoning. The instrument also reads both ways: the same features that flag the common structural configurations AI posts use also locate the rare configurations of human writing.

There are three limitations to our instrument. First, we cover single-pass generation from the five state-of-the-art AI models. Second, the instrument is LLM-run end to end. Like the original study, our validity rests on the validity of Section 7. Third, we tested the robustness of our instrument only against rewording attacks, not against the myriad of other, potentially more sophisticated humanizer and paraphrase tools. Those attack classes remain an interesting avenue for future research (Section 9).

To facilitate this research, we release the pipeline, methodology and code together with the paper.

## 9 Limitations

Scope of the detection claim. Our design measures single-pass generation, both as generated and after rewording (Section 5.4): every AI post is a model’s first-shot response to a reverse-engineered brief, optionally rewritten by the model that generated it - not humanized, heavily post-processed, or human-AI collaborative content.

Brief lossiness. The mirror mechanism is informationally asymmetric: the human author wrote from full business context, while each AI model conditions only on a reverse-engineered brief. Some human-AI structural differences may therefore reflect what the brief fails to carry rather than how models write. We inherit this limitation from the original study.

Publication-year confound (inherent). Our human posts predate ChatGPT and mirrors are generated in 2025-26, so publication period and authorship are confounded. The publication-year check provides evidence that structural features cannot predict when a human post was published, and removing the most year-associated features does not change the headline result. Still, same-period human-vs-AI comparison remains open.

Magnitude interpretation. Every effect we report is consistent in direction with the original and larger in magnitude. The discovery stage optimizes features to separate these six sources on the corpus, and a commercial-native schema fitted to the domain probably amplifies the effect. We therefore do not claim that commercial content is intrinsically more separable than fiction, only that the underlying effect replicates.

Corpus scale and composition. The corpus has 2,250 prompts against the original’s 10,272, and the feature-discovery set is 4.4% of the corpus versus their roughly 1%. Our bootstrap CI widths are adjusted accordingly, but per-class attribution and top-tail rarity statistics are underpowered relative to the original. The sampling corpus is softwareheavy, US-heavy, and 75.5% of post snapshots date from 2020-2022.

Attack coverage. Our rewording test covers rewording by the generating models themselves, following an editing protocol of Chakrabarty et al. (2025). It does not cover a range of humanizer or paraphrase tools. Claims about those attack classes remain open.

Competing interests. The author operates Sitefire, a commercial GEO product (formal statement in the Statements section).

## Statements

Data availability. The release package is at https://github.com/pulse-energyeu/slopshape. Publicly released: the sampling ledger with a deterministic fetch pipeline that rebuilds the exact human corpus from public archives, the template schema with its NarraBench mapping, the complete 214-feature instrument, the complete prompt set (including the rewording-attack prompt), the analysis code (including the fork patch against the original’s released pipeline and the rewording verification gates), and the aggregate artifacts and regeneration scripts behind every number and figure in this paper. Available to researchers on request, under a non-commercial research agreement: the per-post feature answers, deterministic refits of the trained models, and the generated briefs and mirrors. We gate these and only these because the ready-to-use scoring assets are the direct substrate of a commercial product and because the mirrors embed content derived from copyrighted source posts. We do not redistribute the copyrighted human posts themselves, and everything needed to audit the paper’s claims can be rebuilt with the released pipeline.

Ethics. Human posts are publicly published marketing content collected from public web archives and are not redistributed. No personal data beyond published bylines is processed.

Competing interests. The author operates Sitefire, a commercial GEO product. The study’s methods, instrument, prompts, code, and aggregate artifacts are publicly released, with post-level data available to researchers on request, so that every claim can be verified independently of that interest.

Acknowledgments. We thank J. Russell for correspondence resolving ambiguities in the original study’s reported constants.

## References

P. Aggarwal, V. Murahari, T. Rajpurohit, A. Kalyan, K. Narasimhan, and A. Deshpande. 2024. GEO: Generative Engine Optimization. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2024). arXiv:2311.09735.

Ahrefs. 2025. 74% of New Webpages Include AI Content (Study of 900k Pages). Ahrefs Blog. https: //ahrefs.com/blog/what-percentageof-new-content-is-ai-generated/.

Anthropic. 2026. Claude text watermark (product announcement). https://www.anthropic. com/news/claude-text-watermark.

C. Brooks, S. Eggert, and D. Peskoff. 2024. The Rise of AI-Generated Content in Wikipedia. In EMNLP 2024 Workshop on NLP for Wikipedia. arXiv:2410.08044.

T. Brown, B. Mann, N. Ryder, et al. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

T. Chakrabarty, P. Laban, and C.-S. Wu. 2025. Can AI writing be salvaged? Mitigating Idiosyncrasies and Improving Human-AI Alignment in the Writing Process through Edits. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. arXiv:2409.14509.

M. Chen, X. Wang, K. Chen, and N. Koudas. 2025. Generative Engine Optimization: How to Dominate AI Search. arXiv:2509.08919.

T. Chen and C. Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

J. Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

L. Dugan, A. Hwang, F. Trhlik, J. M. Ludan, A. Zhu, H. Xu, D. Ippolito, and C. Callison-Burch. 2024. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. In Proceedings ofACL 2024. arXiv:2405.07940.

Google. 2024. Spam policies for Google web search. https://developers.google. com/search/docs/essentials/spampolicies. See also the accompanying March 2024 scaled-content-abuse update “New ways we’re tackling spammy, low-quality content on Search” (2024-03-05).

Graphite. 2025. More Articles Are Now Created by AI Than Humans. Graphite Five Percent blog. https://graphite.io/five-percent/ more-articles-are-now-created-byai-than-humans.

Graphite. 2026. AI Now Writes as Many Online Articles as Humans. Graphite Five Percent blog. https://graphite.io/fivepercent/ai-now-writes-as-manyonline-articles-as-humans-do.

S. Hamilton, M. Wilkens, and A. Piper. 2025. NarraBench: A comprehensive framework for narrative benchmarking. arXiv:2510.09869.

A. Hans, A. Schwarzschild, V. Cherepanova, H. Kazemi, A. Saha, M. Goldblum, J. Geiping, and T. Goldstein. 2024. Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. In Proceedings ofthe 41st International Conference on Machine Learning (ICML 2024). arXiv:2401.12070.

M. Khosravi and H. Yoganarasimhan. 2026. Impact of AI Search Summaries on Website Traffic: Evidence from Google AI Overviews and Wikipedia. arXiv:2602.18455.

D. Kobak, R. González-Márquez, E.-Á. Horvát, and J. Lause. 2025. Delving into LLM-assisted writing in biomedical publications through excess vocabulary. Science Advances, 11. arXiv:2406.07016.

K. Krippendorff. 2004. Content Analysis: An Introduction to Its Methodology, 2nd edition. Sage Publications.

K. Krishna, Y. Song, M. Karpinska, J. Wieting, and M. Iyyer. 2023. Paraphrasing evades detectors of AIgenerated text, but retrieval is an effective defense. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023). arXiv:2303.13408.

W. Liang et al. 2025. The widespread adoption of large language model-assisted writing across society. Patterns (Cell Press). arXiv:2502.09747.

S. M. Lundberg and S.-I. Lee. 2017. A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017).

E. Masrour, B. Emi, and M. Spero. 2025. DAMAGE: Detecting Adversarially Modified AI Generated Text. arXiv:2501.03437.

A. Mathur, C. Liu, K. Tan, and Y. Liu. 2026. Scalable Detection of Adversarial Synthetic Slop and Coordinated Media Abuse: A LoRA-Enabled Multimodal Defense System. Google Research. https://storage.googleapis.com/ gweb-research2023-media/pubtools/ 1039291.pdf.

E. Mitchell, Y. Lee, A. Khazatsky, C. D. Manning, and C. Finn. 2023. DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. In Proceedings of the 40th International Conference on Machine Learning (ICML 2023). arXiv:2301.11305.

M. Perkins, J. Roe, et al. 2024. GenAI Detection Tools, Adversarial Techniques and Implications for Inclusivity in Higher Education. arXiv:2403.19148.

J. Russell, M. Karpinska, D. Akinode, J. Zhou, K. Thai, B. Emi, M. Spero, and M. Iyyer. 2026b. AI use in American newspapers is widespread, uneven, and rarely disclosed. In Proceedings of ACL 2026. arXiv:2510.18774.

J. Russell, P. Rajendhran, C. Pham, M. Iyyer, and J. Wieting. 2026a. StoryScope: Investigating idiosyncrasies in AI fiction. arXiv:2604.03136.

V. S. Sadasivan, A. Kumar, S. Balasubramanian, W. Wang, and S. Feizi. 2023. Can AI-Generated Text be Reliably Detected? arXiv:2303.11156.

C. Shaib, T. Chakrabarty, D. Garcia-Olano, and B. C. Wallace. 2025. Measuring AI “Slop” in Text. arXiv:2509.19163.

B. Thompson, M. P. Dhaliwal, P. Frisch, T. Domhan, and M. Federico. 2024. A Shocking Amount of the Web is Machine Translated. In Findings ofACL 2024. arXiv:2401.05749.

Z. Tian, Y. Chen, Y. Tang, J. Liu, and R. Jia. 2026. Diagnosing and Repairing Citation Failures in Generative Engine Optimization. arXiv:2603.09296.

E. P. Torrance. 1966. Torrance Tests of Creative Thinking. Personnel Press.

![](images/77a346146eae45ec920d2ff20444f009478ce6e332ab85e6e4f0782e6cc40f7b.jpg)  
Figure A1. Word-count distributions by source (boxes = IQR, whiskers to 1.5 IQR, outliers hidden). Four of the five AI models produce longer posts than their human sources on average, while kimi-k2.5 runs slightly shorter.

B. Warner, A. Chaffin, B. Clavié, O. Weller, O. Hallström, S. Taghadouini, A. Gallagher, R. Biswas, F. Ladhak, T. Aarsen, N. Cooper, G. Adams, J. Howard, and I. Poli. 2024. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. arXiv:2412.13663.

D. Weber-Wulff et al. 2023. Testing of detection tools for AI-generated text. International Journalfor Educational Integrity, 19(26).

P. Zhang, R. Cui, and D. J. Zhang. 2026. The Impact of AI Search on the Online Content Ecosystem: Evidence from Google and Reddit. arXiv:2605.16428.

## A Data collection detail

This appendix carries the corpus composition, mirror coverage, and funnel mechanics summarized in Section 3.

Of the 306 selected domains, 264 yielded usable posts (40 had no fetchable qualifying articles, 2 were emptied by the content filters), and 4 replacement domains drawn from the same qualification pool during the fetch gave the 268 corpus domains. A per-domain cap of 20 posts was applied (four domains reach at most 29). The largest single domain contributes 1.3% of the corpus.

## B Schema and NarraBench mapping

This appendix maps each NarraBench dimension (Hamilton et al., 2025) to its counterpart in the commercial schema of Section 4.1.

## C Instrument detail

This appendix carries the final instrument’s composition, the quality-gate detail behind Section 4.3, and the scoring-mode gate behind Section 4.4.

Quality gate. The gate never sees human/AI labels, and its dominant rejection defect was nonexhaustive value lists. The gate prompt and all rejection reasons are preserved in the release. The deduplication cutoff sweep is check 4 in Appendix G.

Aspect-based application check. Scoring one dimension at a time versus scoring a whole post in one call, on 12 posts: 99.97% vs 99.75% of items answered, cross-mode agreement 0.774 (original: 95.4% vs 68.4%, with the single-call mode systematically dropping later dimensions). We kept per-dimension scoring, per the original.

## D Protocol detail

This appendix records the executed hyperparameter grids and the hyperparameter sensitivity result behind Section 4.7.

The binary task’s grid comprised 108 configurations over n\_estimators, max\_depth, reg\_lambda, and scale\_pos\_weight, and the six-source task ran its own 27-configuration grid, both centered on the original’s published constants (500/7/1.0). Hyperparameter sensitivity: refitting with those constants, untuned, gives binary 97.8 (AUPRC 1.000) and sixsource macro-F1 79.3 (accuracy 79.5) - within 0.2 points of the tuned configurations on both tasks.

## E Full deviation register

This appendix carries the full deviation register with each deviation’s defense.

## F Additional results

This appendix carries the additional detection baselines, the LDA figure, the rarity tail table, the model-specific feature table, and the geometry numbers.

Protocol notes: the feature-variant models and the stylometric, TF-IDF, and length baselines are retrained on train+val per the original’s protocol. The ModernBERT row (single seed, their 512-token 3- epoch configuration) and the Binoculars-style row (Qwen2.5-0.5B base/instruct pair in place of their Falcon-7B pair) are single fine-tunes or zero-shot substitutes, disclosed.

Cohen’s d uses the unweighted root-meansquare of the two group SDs (the original pooled by group size).

Selection is by per-class SHAP concentration, with the full criteria in the release. The original selected 75 features carrying 90 per-source values (counts 32/26/11/11/7/3).

<table><tr><td>Axis</td><td>Shares (%)</td></tr><tr><td>Verticals</td><td>software/SaaS 26.5, e-commerce 24.3, services 17.4, fintech 15.7, devtools 11.7, health 3.8, edtech 0.6</td></tr><tr><td>Source frame</td><td>YC 57.2, Inc5000 33.7, G2 6.8, FT1000 2.3</td></tr><tr><td>Snapshot years</td><td>2008-17: 8.5, 2018-19: 16.0, 2020-21: 41.9, 2022: 33.6</td></tr></table>

Table A1 - Data composition.

<table><tr><td>Source</td><td>Posts</td><td>Mean words</td><td>Within 10% of source length</td></tr><tr><td>human</td><td>2,250</td><td>1,186</td><td></td></tr><tr><td>gpt-5.4</td><td>2,250</td><td>1,541</td><td>.07</td></tr><tr><td>deepseek-v3.2</td><td>2,250</td><td>1,447</td><td>.16</td></tr><tr><td>gemini-3-flash</td><td>2,250</td><td>1,283</td><td>.41</td></tr><tr><td>claude-sonnet-4.6</td><td>2,250</td><td>1,279</td><td>.66</td></tr><tr><td>kimi-k2.5</td><td>2,250</td><td>1,064</td><td>.47</td></tr></table>

Table A2 - AI mirror coverage and length. The last column reports, per model, the share of mirrors whose word count lands within 10 percent of their human source’s word count.

![](images/306c84d35e27e6ca2af3e2d4027ade3412179afc810f68749ff665117b195dbf.jpg)  
Figure F2. Two-component LDA projection of the encoded structural feature space (12,900 posts). Human posts separate along LD1, while the five AI models overlap heavily with each other, consistent with the original’s geometry finding.

Geometry. The average distance from a human post to its ten nearest neighbors in the encoded structural feature space is 1.43 times larger than for an AI post, the human class as a whole is 1.42 times more dispersed than the average AI class, and the centers of the human and AI point clouds sit clearly apart (centroid distance 11.7 in z-scored units). The original found the same picture in fiction - humans more dispersed, the AI models clustered together (+22% dispersion, 1.13x neighbor radius) - and our version of it is again larger.

## G Robustness checks, full table

This appendix carries the full results and caveats behind the check summary of Table 8. Every check runs under the original’s protocol (final models retrained on train+val).

## H Instrument validity, full detail

This appendix carries the style-boundary audit behind Section 7.

Style-boundary audit. On 40 sampled features, human agreement with the model’s binary style/non-style call: annotator A 90.0% (kappa 0.765), annotator B 97.5% (0.936), against a 0.75 bar fixed before the audit. Human-human agreement is 92.5% (0.827). The strict style boundary (GPT-5.4, 3 runs, 0.989 inter-run agreement, 34 exclusions all inside the Writing-Style dimension) is endorsed by the human annotators.

## I Original released-code defects and fixes

This appendix records the five defects we found in the original’s released code. Each is fixed in our fork toward the paper’s stated method, which makes the replication more faithful, not less. The unmodified originals and the full defect record are preserved in the release for side-by-side verification.

Beyond these fixes and authentication/plumbing changes (provider base URLs and API-key paths, a 391-line fork patch over 7 files), no feature, encoding, classifier, or analysis code was modified.

## J Rewording test detail

This appendix carries the rewriting protocol, the verification gates, and the per-model detail behind Sections 4.8 and 5.4.

Prompt design. The rewriting prompt follows the LAMP taxonomy of Chakrabarty et al. (2025): the seven categories of AI-writing artifacts their study identified with professional editors, together with their editing instructions. The prompt is instruction-only (deviation D15). A contentpreservation clause instructs the model to keep every claim, fact, and link and to reword freely within them (deviation D16). The full prompt text is in the release.

<table><tr><td>NarraBench dimension</td><td>Status</td><td>Commercial schema mapping</td></tr><tr><td>Agent</td><td>ADAPTED</td><td>Voices and Sources: authors, experts, brands, customers, readers as discourse roles</td></tr><tr><td>Social Network</td><td>REPLACED</td><td>Voices and Sources: source arrangements and reader interaction, not interpersonal plot relations</td></tr><tr><td>Event</td><td>ADAPTED</td><td>Audience/Problem/Stakes and Timeliness: business triggers, policy changes, lifecycle moments</td></tr><tr><td>Plot</td><td>REPLACED</td><td>Structure and Flow: movement from problem or context to explanation, recommendation, resolution</td></tr><tr><td>Structure</td><td>ADAPTED</td><td>Structure and Flow plus Page Format and Navigation</td></tr><tr><td>Setting</td><td>ADAPTED</td><td>Audience/Problem/Stakes: the reader&#x27;s professional, market, regulatory, technical context</td></tr><tr><td>Time</td><td>KEPT</td><td>Timeliness: chronology, temporal reference, change, timing relevance</td></tr><tr><td>Revelation</td><td>REPLACED</td><td>Explanation Depth plus Evidence and Proof: staged explanation and substantiation</td></tr><tr><td>Perspective</td><td>ADAPTED</td><td>Voices and Sources: institutional, founder, expert, interviewer, direct-reader viewpoints</td></tr><tr><td>Style</td><td>KEPT</td><td>Writing Style: the single mandated style bucket</td></tr></table>

Table B1 - NarraBench mapping (kept / adapted / replaced).
<table><tr><td>Dimension</td><td>Features</td><td>Types (categorical/binary/ordinal/scale/multi)</td><td>Example feature</td></tr><tr><td>Actionability</td><td>34</td><td>18/6/5/1/4</td><td>Conditional Logic Presence</td></tr><tr><td>Writing style</td><td>27</td><td>14/4/4/4/1</td><td>Dominant Sentence Complexity</td></tr><tr><td>Audience</td><td>23</td><td>8/7/4/4/0</td><td>Opening Reader Question Invitation</td></tr><tr><td>Explanation</td><td>21</td><td>13/4/2/1/1</td><td>Caveat Presence</td></tr><tr><td>Timeliness</td><td>20</td><td>11/1/4/2/2</td><td>Primary Change Story</td></tr><tr><td>Voices</td><td>20</td><td>9/4/4/1/2</td><td>Primary Reader Role</td></tr><tr><td>Structure and flow</td><td>19</td><td>10/4/3/1/1</td><td>Primary Repeated Building Block</td></tr><tr><td>Evidence</td><td>17</td><td>5/6/3/1/2</td><td>Attribution Density</td></tr><tr><td>Commercial integration</td><td>15</td><td>5/5/3/2/0</td><td>Proof of Capability Presence</td></tr><tr><td>Page format</td><td>9</td><td>4/3/2/0/0</td><td>Author Identity Treatment</td></tr><tr><td>Purpose</td><td>9</td><td>7/2/0/0/0</td><td>Dominant Communicative Purpose</td></tr></table>

Table C1 - Final 214-feature instrument composition.

Verification gates. All gates were fixed before generation, and all passed. Length drift between a rewritten post and its source was bounded to the ratio interval [0.6, 1.4], with 0 violations (mean ratio 0.922, minimum 0.607). Refusals: 0 of 1,450. Trivial-copy flags: 0 after reruns. For claim preservation, we judged all 1,450 original-rewritten pairs in full: 90.3% preserved every claim before quality control, and after a two-pass regeneration loop (192 regenerations) the full census reads 98.2%, with 26 single-item residuals disclosed and left in.

Rescoring and evaluation. The rewritten posts are scored by the identical frozen stage-5 scorer and prompts (byte-identity asserted), covering all 15,950 answers (1,450 posts x 11 dimensions) with 0 unresolved failures and a rate of answers outside the predefined options of 0.082% on average (0.089% in our main scoring run). The encoding is identical, the classifiers are deterministic refits of the final models, and nothing is retrained on rewritten text. Of the 1,450 reworded AI posts, the structural classifier misclassifies 5 as human-written, the style-only classifier 16, and the all-features classifier 4. Table J1 gives the attack magnitude and the attacked structural score per model.

<table><tr><td>#</td><td>Deviation from the original</td><td>Defense</td></tr><tr><td>D1</td><td>Domain: B2B blog posts</td><td>The research question</td></tr><tr><td>D2</td><td>Brief extractor gemini-3-flash (their gemini-2.5-flash is deprecated)</td><td>Declared</td></tr><tr><td>D3</td><td>Brief prompt adapted, anti-quotation clause added</td><td>Publisher-blind</td></tr><tr><td>D4</td><td>Dedup cutoff 0.85 retained from the original (not re-selected)</td><td>Parity, sweep published (Appendix G, check 4)</td></tr><tr><td>D5</td><td>Style audit 3 runs with agreement statistics (theirs: 1 run)</td><td>Exceeds the original</td></tr><tr><td>D6</td><td>Domain-disjoint splits, single-unblinding holdout (theirs: random</td><td>Stricter</td></tr><tr><td>D7</td><td>prompt split) Rarity metric re-implemented from text</td><td>Verified on their data, magnitudes disclaimed</td></tr><tr><td>D8</td><td>Extraction and discovery via OpenAI direct API</td><td>Provider plumbing</td></tr><tr><td>D9</td><td>Encoder fixes (ordinal by taxonomy position, nominal one-hot)</td><td>Their released code deviated from their own paper</td></tr><tr><td>D10</td><td>Corpus scale ~2,250 prompts (~1/4.5 of theirs), feature-discovery set</td><td>Budget and domain scarcity of archived pre-2022 posts, power note in Section 9</td></tr><tr><td>D11</td><td>4.4% of corpus vs their ~1% Commercial-native template schema discovered from human posts (theirs: NarraBench fiction schema)</td><td>Fiction lens misfits commercial content, mechanism unchanged, mapping table</td></tr><tr><td>D12</td><td>Mirror max_tokens 8,000 (theirs: 128k/65k)</td><td>published Matched to commercial target lengths</td></tr><tr><td>D13</td><td>Evaluation-stage models upgraded within family lineage, the five AI models unchanged</td><td>Same-variant newer versions for non-generating tasks</td></tr><tr><td>D14</td><td>Rewording test: each of the five AI models rewrites its own posts, all 1,450 test mirrors (theirs: 278 posts, one model as rewriter)</td><td>Realistic production pattern, removes the original&#x27;s single-model asymmetry, yields a per-model breakdown</td></tr><tr><td>D15</td><td>Rewording prompt is instruction-only (the original&#x27;s 25 few-shot professional examples are fiction rewrites)</td><td>Domain adaptation, declared - not parity</td></tr><tr><td>D16</td><td>Content-preservation constraint added (keep claims, facts, links)</td><td>Verified by a full claim-preservation census (Appendix J)</td></tr></table>

Table E1 - Deviation register (full defenses).

<table><tr><td>Variant / baseline</td><td>n feat</td><td>Test macro-F1</td><td>AUPRC</td><td>Original</td></tr><tr><td>Binoculars-STYLE (small-pair substitute, context only)</td><td></td><td>78.2</td><td>.980</td><td>(their true Binoculars: 55.9 / .404)</td></tr><tr><td>Length-only logistic</td><td>1</td><td>45.5</td><td>.871 (prevalence baseline .833 at the 5:1 ratio)</td><td>55.9</td></tr></table>

Table F1 - Additional binary detection baselines (rows not in Table 3).

<table><tr><td>Tail</td><td>Human posts</td><td>AI posts (five models pooled)</td><td>Human share of tail</td><td>Original (test basis)</td></tr><tr><td>Rarest 1%</td><td>149</td><td>4</td><td>97.4%</td><td>42 human vs 41 AI</td></tr><tr><td>Rarest 5%</td><td>592</td><td>80</td><td>88.1%</td><td></td></tr><tr><td>Rarest 10%</td><td>1,026</td><td>305</td><td>77.1%</td><td></td></tr></table>

Table F2 - Rarity tail composition.

<table><tr><td>Source</td><td>Model-specific features (up to 5)</td></tr><tr><td>human</td><td>Execution Support Resources (ACT_CHK_025); Decision Rule Presence (ACT_STP_006); False Default Challenged (AUD_PRB_005); Stakes Escalation (AUD_STK_005); External Voice</td></tr><tr><td>gpt</td><td>Construction (EVD_MIX_005) Artifact Production Requirement (ACT_CON_015); Limits or Caveats Presence (EVD_MTH_002); Numeric Density (EVD_NUM_001); Baseline Disclosure for Comparative Numbers (EVD_NUM_004);</td></tr><tr><td>claude gemini</td><td>Quantitative Claim Sourcing (EVD_NUM_006) Numbered Procedure Presence (ACT_STP_004)</td></tr><tr><td>deepseek kimi</td><td>Self-Reference Frequency (COM_ROL_005); Evidence Limitation Disclosure (EXP_KNW_009) Operational Threshold Use (EVD_NUM_005); Change Requested of Reader (PUR_JOB_004)</td></tr></table>

Table F3 - Model-specific features by source (top features by six-source SHAP concentration). The table lists up to five features per source, with per-source totals in Section 6.

<table><tr><td>#</td><td>Check</td><td>Result</td></tr><tr><td>1</td><td>Length confound (original&#x27;s Section F)</td><td>Length-matched test subsample (1,545 posts, median words human 1,004 vs AI 1,081): headline 98.1 vs 98.0 unmatched - unchanged, mirroring the original (93.2 to 93.2). Rarity d on the matched subset 1.87, corr(words, rarity) = 0.09. Per-model matched rarity means preserve the full-corpus ordering (deepseek 0.54 remains</td></tr><tr><td>2</td><td>Publication-year check</td><td>closest to humans) Predicting a human post&#x27;s publication period from structural features performs at chance (macro-F1 49.0, grouped CV). Removing the most year-associated features leaves the headline flat (97.8-98.0). Top-25 year/authorship feature overlap 4/25 -</td></tr><tr><td>3</td><td>Template-vs-direct discovery (the original&#x27;s template-necessity ablation)</td><td>publication year is not the signal Same candidate yield (raw-on-text 354 vs template-run mean 356.7, all 11 dimensions) but only ~23% bidirectional semantic overlap at 0.85 embedding similarity - the discovery pathway substantially shapes which features are found (original: 6 of top-20 overlap). The raw feature set was not scored at scale (declared)</td></tr><tr><td>4</td><td>Deduplication cutoff sweep</td><td>Cutoffs 0.70-0.95 published: 200/233/251/266/277/282 features. Silhouette monotonically decreasing (0.182 at 0.70 to 0.065 at 0.85), so silhouette alone would prefer lower cutoffs - divergence disclosed. 0.85 retained for parity (merge 5.7% vs</td></tr><tr><td>5</td><td>Feature-discovery-set domain overlap</td><td>their 25.5%), sensitivity low 11 of the 82 feature-discovery-set domains land in the test split. Sensitivity excluding them: 98.2</td></tr><tr><td>6</td><td>Memorization (original&#x27;s Section E, both rules, exact implementations)</td><td>Exact 13-gram rule: 0.19% of the 10,750 human-mirror pairs (the 2,150 classification prompts x 5 mirrors) share at least one 13-gram (shuffled-human control: 0.0%, original: 0.70% flagged). Near-verbatim rule (8-gram coverage &gt;= 5%, &gt;= 4 distinct 8-grams, longest common span &gt;= 30 tokens): 0 pairs. Excluding all 16 affected prompts: headline 97.9 (delta -0.1 points, original&#x27;s filtered deltas &lt;= +0.3)</td></tr><tr><td>7</td><td>Vertical heterogeneity (Kruskal-Wallis, original&#x27;s topic robustness)</td><td>H = 8.997, p = 0.109 over the 6 verticals with &gt;= 20 test posts - not significant, consistent with the original&#x27;s null (H = 4.69, p = 0.46). Per-vertical macro-F1 ranges from 96.5 (services) to 100 (software, devtools, health). Caveats: grouping uses &gt;= 20 test posts where the original used &gt;= 20 prompts (the thinnest vertical rests on 5 prompts), and per-post correctness observations are domain-clustered, an</td></tr><tr><td>8</td><td>Learning curve</td><td>independence violation the original&#x27;s version shares Train+val fractions 25/50/75/100%: 96.8 / 97.8 / 98.0 / 98.0 (3 seeds each) - performance saturates by ~75%, so corpus size is not the binding constraint on the headline</td></tr></table>

Table G1 - Robustness checks, full results.

<table><tr><td>#</td><td>Defect in released code</td><td>Symptom</td><td>Our fix</td></tr><tr><td>B1</td><td>Stage config keys passed twice to the provider layer</td><td>Every stage runner crashes on launch</td><td>De-duplicated the keyword pass-through</td></tr><tr><td>B2</td><td>Stage 5 only recognizes a human_story source column</td><td>All human posts silently dropped: the feature matrix would contain zero human rows</td><td>Source-column handling accepts the corpus&#x27;s human and mirror columns</td></tr><tr><td>B3</td><td>JSON-generation path ignores the thinking configuration</td><td>The paper states stage 5 runs “minimal thinking&quot;, but the released code sets no thinking budget and runs full reasoning (83x slower per dimension call)</td><td>Thinking budget set as a stage parameter, matching the paper&#x27;s stated method</td></tr><tr><td>B4</td><td>Shipped stage-3 prompt does not implement the paper&#x27;s stated method</td><td>The prompt asks for a two-template quality comparison and contains {group_a_json}/{group_b_json} placeholders the code never fills, while the paper describes all six templates presented together with cross-source</td><td>Prompt rewritten to the paper&#x27;s specification (all templates together, structured per-source notes, divergences, executive summary). The shipped prompt and the first defective run are preserved</td></tr><tr><td>B5</td><td>Released encoding deviates from the paper&#x27;s stated scheme (deviation D9)</td><td>Ordinal and nominal features not encoded as the paper describes</td><td>Ordinal encoded by taxonomy position, nominal one-hot, per the paper&#x27;s text</td></tr></table>

Table I1 - Released-code defects and fixes.

<table><tr><td>AI model</td><td>Surviving 13-gram share</td><td>Attacked structural macro-F1</td></tr><tr><td>gpt-5.4</td><td>0.175</td><td>98.1</td></tr><tr><td>claude-sonnet-4.6</td><td>0.535</td><td>98.1</td></tr><tr><td>gemini-3-flash</td><td>0.003</td><td>98.1</td></tr><tr><td>deepseek-v3.2</td><td>0.489</td><td>98.1</td></tr><tr><td>kimi-k2.5</td><td>0.141</td><td>97.6</td></tr></table>

Table J1 - Per-model attack magnitude and attacked structural detection. The surviving 13-gram share is the fraction of a post’s 13-word sequences that still appear verbatim after rewriting (mean 0.269 across models, i.e. 73% of 13-word sequences replaced on average).