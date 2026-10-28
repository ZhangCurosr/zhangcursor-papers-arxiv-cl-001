# Text Generation in Low-Resource Languages Using Human-Curated Vocabularies

Michael Christlambert Sinanta<sup>1\*</sup> Muhammad Dehan Al Kautsar<sup>2\*</sup> Amanda Nurul Izzah<sup>1</sup> Thanh-Nhi Nguyen<sup>1</sup> Phan Phuong Mai Chau<sup>2</sup> Zheng Wei Lim<sup>3</sup> Fajri Koto<sup>2</sup>

<sup>1</sup>Independent Researcher <sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence <sup>3</sup>University of Amsterdam

michaelchristlambert@gmail.com, muhammad.dehan@mbzuai.ac.ae

## Abstract

Synthetic text generation offers a promising approach to alleviating data scarcity in lowresource languages. However, its effectiveness is inherently limited for languages that are scarcely represented, or entirely absent, in an LLM’s pretraining data. At the same time, many such languages have human-curated vocabularies that provide rich lexical and semantic information, often accompanied by grammatical information such as part-of-speech labels. In this work, we ask whether LLMs can leverage these resources to generate coherent text in extremely low-resource and unseen languages. We introduce a framework that combines human-curated vocabularies with a small number of human-written exemplars that provide syntactic signals, while sampling candidate words according to a Zipfian distribution. Across languages with varying levels of resource availability, we evaluate the generated text using lexical diversity, off-target ratio, and human judgments of fluency. Our results show that reasoning-enabled LLMs can generate fluent text even in extremely low-resource lan guages, with particularly strong results at the sentence level and promising performance at the paragraph level. Our findings highlight a practical pathway for bootstrapping synthetic data in low-resource languages, expanding the amount of usable text even when existing digi tal resources are extremely limited.<sup>1</sup>

## 1 Introduction

More than 7,000 languages are spoken worldwide, yet only a small fraction are represented in the textual resources commonly used to train language models. For example, Wikipedia and Common Crawl contain textual data for only about 4% of the world’s languages, while human-curated lexicons and word lists cover approximately 70% (Wang et al., 2022). This broader coverage reflects longstanding language documentation efforts, through which linguists and language communities have compiled human-curated vocabularies that record words, their meanings, and other linguistic information. Their broad coverage creates an opportunity to leverage these existing resources to expand text resources for low-resource languages.

![](images/b99fa7097dac50ce88bb1a59cc2b82e66fac3b2b433ca0ab34d0fcca6933148b.jpg)  
Figure 1: Overview of our vocabulary-guided synthetic text generation pipeline. Human-curated vocabulary is sampled using an open-source word-frequency list, while we then sample the high- and low-frequency words using our method described in Section 3. We then combine it with few-shot human-written exemplars to construct the generation prompt.

One promising way to realize this opportunity is through synthetic text generation. LLMs are increasingly used to generate synthetic data for pretraining and post-training (Wang et al., 2023; Doshi et al., 2024; Anikina et al., 2025; Stenlund et al., 2026). However, such generation largely relies on the linguistic knowledge already encoded in LLMs, limiting its effectiveness for languages that are poorly represented in their pretraining data (Guo et al., 2024; Li et al., 2025). Human-curated vocabularies offer a complementary source of linguistic knowledge, yet their potential for LLMbased synthetic text generation remains largely unexplored (Yong et al., 2024). This idea has a natural parallel in human language learning: when learning an unfamiliar language, people consult dictionaries to acquire new words and use examples to learn how those words appear in context. Inspired by this process, we ask: Can LLMs leverage human-curated vocabularies to generate coherent and lexically diverse text in low-resource languages, including extremely underrepresented languages?

Our contributions are threefold:

1. We introduce a training-free framework that leverages LLM reasoning to generate text in low-resource languages from human-curated vocabularies and a small number of humanwritten exemplars. Our framework adapts from Zipfian sampling to expose the model to a broad yet natural distribution of vocabulary during generation.

2. Across six languages spanning different levels of resource availability, our framework consistently improves lexical diversity and reduces off-target generation over practical baselines. Human evaluation further shows improved fluency, with particularly strong results for the extremely low-resource Acehnese and Hitu.

3. Through extensive analyses, we examine when and why vocabulary-guided generation works, studying the impact of vocabulary quality, reasoning versus non-reasoning generation, and increasing generation volume. Our results highlight vocabulary quality as a key consideration and show whether gains in lexical diversity persist as synthetic data are generated at larger scale.

## 2 Related Works

Synthetic Text Generation with LLMs. Synthetic data has long been used to supplement scarce human-written data in NLP (Feng et al., 2021). Early augmentation methods typically transform existing examples through operations such as synonym replacement, random insertion and deletion, and word swapping (Wei and Zou, 2019; Karimi et al., 2021). The emergence of LLMs has enabled a shift toward generating new synthetic text at scale, with applications ranging from augmenting pretraining corpora (Doshi et al., 2024; Zhu et al., 2026) to constructing post-training resources such as instruction-tuning data (Wang et al., 2023). Despite their strong generation capabilities, LLMs tend to produce text with lower lexical diversity than human-written data (Guo et al., 2024; Kambhatla et al., 2025). This limitation is particularly relevant to our goal of expanding the lexical coverage of languages with limited textual resources. Therefore, we treat lexical diversity, alongside fluency, as a central dimension of synthetic text quality.

Synthetic Data for Low-Resource Languages. Synthetic text generation is particularly challenging for low-resource languages, where LLMs have limited linguistic knowledge. Generation quality tends to deteriorate as language resources decrease (Bafna et al., 2025), and models may drift into higher-resource languages when generating underrepresented ones (Marchisio et al., 2024). Nevertheless, LLM-generated data have been explored for low-resource applications ranging from question answering to story generation (Namboori et al., 2023; Cegin et al., 2025; Pranida et al., 2025), with target-language demonstrations shown to improve their utility (Anikina et al., 2025). These approaches, however, largely rely on linguistic knowledge already encoded in the LLM. Our work instead asks whether LLMs can generate fluent and lexically diverse text with minimal off-target generation in extremely underrepresented languages, including languages not seen during LLM pretraining, where we leverage LLM reasoning to combine with external lexical knowledge to provide additional linguistic information at inference time.

Human-Curated Vocabularies in NLP. Humancurated lexical resources, including dictionaries, bilingual lexicons, and word lists, cover substantially more languages than commonly available text corpora (Wang et al., 2022). Prior work has leveraged these resources for lexical augmentation, word-for-word data synthesis, and machine translation (Feng et al., 2021; Wang et al., 2022; Peng et al., 2020). Recent studies further show that providing dictionaries or grammatical resources in context can help LLMs translate languages scarcely represented in pretraining (Tanzer et al., 2024; Zhang et al., 2024). These approaches, however, primarily focus on translation. Closest to our work, Yong et al. (2024) condition LLMs on bilingual lexicons to generate task-specific data, which are subsequently translated word by word into the target language. In contrast, we study open-ended generation directly in the target language. We use human-curated vocabularies as lexical knowledge and a few human-written exemplars as syntactic guidance, requiring the LLM to compose new text rather than translate or modify existing sentences.

## 3 Methodology

We illustrate our training-free, vocabulary-guided synthetic text generation pipeline in Figure 1. The pipeline consists of four stages: (1) collecting human-curated dictionaries and word-frequency information for each target language, (2) collecting a small set of human-written exemplars in sentences and paragraphs, (3) sampling candidate words according to their frequency distribution, and (4) constructing the generation prompt from the sampled words and exemplars.

Stage 1: Collecting Dictionaries and Estimating Word Frequencies. We first collect a humancurated dictionary D for each target language and standardize its entries into a common format. Each entry $w \in \mathcal { D }$ consists of a target-language word, its English equivalent en(w), an English definition, a part-of-speech (POS) tag, and, when available, an example sentence illustrating its usage. Because the available resources differ substantially across languages and formats, we use languagespecific procedures to construct the dictionaries. Appendix A describes the data sources and processing procedures and reports the number of entries collected for each language (Table 5).

We next estimate how frequently each dictionary word is likely to occur in natural text. Direct frequency estimates are unavailable or unreliable for many of our target languages, particularly the extremely low-resource ones. We therefore derive approximate frequencies from English using the Dictionary Word Frequency dataset,<sup>2</sup> which provides unigram counts from Google Books. Each entry w is assigned the frequency $f ( w )$ of its English equivalent en(w). Entries whose English equivalent has no frequency are excluded from frequency-based sampling, and the remaining entries form the vocabulary $\nu \subseteq \mathcal { D }$ . We rank the entries of V from highest to lowest $f ( w )$ and denote the rank of entry w by $r ( w )$ , so that $r ( w ) = 1$ for the most frequent entry. The resulting distribution is heavy-tailed, with a small number of highly frequent words and a long tail of less frequent words, consistent with Zipf’s law. We use this ranking for word-candidate sampling in Stage 3.

Stage 2: Collecting Few-Shot Exemplars. In parallel, we collect a small set of human-written exemplars to provide syntactic and contextual guidance for generation (Anikina et al., 2025). For each language (Section 4), we recruit one native-speaker annotator. Recruiting native speakers is particularly challenging for the extremely low-resource languages in our study. To ensure the quality of the exemplars, we apply strict eligibility criteria: annotators must have at least a senior high school education, must have lived for at least 10 years in a community where the target language is spoken, and must have at least one parent who speaks the language. Each annotator writes 40 sentences and 5 paragraphs. The sentences cover four target lengths (5, 10, 15, and 20 words), with 10 sentences per length. Each paragraph contains at least three sentences and at least 25 words. We denote the resulting pool of exemplars by E. Each exemplar $e \in { \mathcal { E } }$ consists of a target-language text, its English translation, and the set of dictionary entries $\mathcal { D } ( e ) \subseteq \mathcal { D }$ used to compose it, following the schema of Stage 1. These exemplars show how dictionary entries are composed naturally in context, and we use them as few-shot guidance during generation. Appendix B describes the annotation procedure in detail.

Stage 3: Sampling Word Candidates. To generate each synthetic sentence or paragraph, we provide the LLM with two inputs: s few-shot exemplars and k word candidates. The exemplars provide syntactic and contextual guidance, while the word candidates encourage lexical diversity. Sampling uniformly from the vocabulary, however, would overrepresent rare words and underrepresent common ones, including function words. We therefore sample the word candidates using the frequency estimates from Stage 1, so that they better reflect natural patterns of word usage.

We split V at a rank boundary b into highfrequency entries H and low-frequency entries $\mathcal { L }$ (Figure 1):

$$
\mathcal { H } = \{ w \in \mathcal { V } : r ( w ) \leq b \} , \qquad \mathcal { L } = \mathcal { V } \setminus \mathcal { H } .\tag{1}
$$

The word candidates should include entries from both sets. A composition parameter $c \in [ 0 , 1 ]$ controls the proportion of high-frequency entries:

of the k word candidates, $k _ { H } = \lfloor c \cdot k \rfloor$ come from H and $k _ { L } = k - k _ { H }$ from ${ \mathcal { L } } .$

Sampling low-frequency entries independently tends to yield unrelated words that are hard to combine into a coherent text. We therefore restrict them to a single topic so that they are semantically related. For each generation, we draw a topic t uniformly from a fixed list of 20 topics.<sup>3</sup> We compute the semantic similarity sim(w, t) between the English equivalent en(w) of each entry $w \in { \mathcal { L } }$ and the topic t using a sentence-transformer model,<sup>4</sup> precomputing these scores once per topic. We then define $\mathcal { L } _ { t } \subseteq \mathcal { L }$ as the $\lceil \rho \cdot \vert { \cal C } \vert \rceil$ entries with the highest similarity to $t ,$ where $\rho \in ( 0 , 1 ]$ is the topic fraction.

Within a subset $S \in \{ \mathcal { H } , \mathcal { L } _ { t } \}$ , each entry is sampled with probability proportional to its estimated frequency:

$$
p ( w \mid S ) = { \frac { f ( w ) } { \sum _ { w ^ { \prime } \in S } f ( w ^ { \prime } ) } } , \qquad w \in S .\tag{2}
$$

The candidate set C is then

$$
C = C _ { \mathcal { H } } \cup C _ { \mathcal { L } _ { t } } , \qquad | C | = k ,\tag{3}
$$

where $C _ { \mathcal { H } }$ consists of $k _ { H }$ entries drawn from H and $C _ { \ * { c _ { t } } }$ of $k _ { L }$ entries drawn from $\mathcal { L } _ { t }$ , both without replacement according to Equation (2), renormalized after each draw.

Stage 4: Assembling the Prompt. Given k and s, we first draw s exemplars $e _ { 1 } , \ldots , e _ { s }$ uniformly at random without replacement from $\mathcal { E }$ . Each exemplar $e _ { j }$ is paired with its corresponding word candidate set $C _ { j } = \mathcal { D } ( e _ { j } )$ , where $\mathcal { D } ( e _ { j } )$ contains the dictionary entries provided when $e _ { j }$ was created (Stage 2). Importantly, each exemplar uses only a subset of the provided candidates, demonstrating that the model should select words that fit naturally in context rather than force all candidates into the generated text. We then sample a new candidate set C following Stage 3 and construct the prompt as

$$
P = \big [ ( C _ { 1 } , e _ { 1 } ) ; \ldots ; ( C _ { s } , e _ { s } ) ; C ; I \big ] ,\tag{4}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation and I instructs the model to generate a new text using suitable words from C. We submit $P$ to an LLM with reasoning enabled, allowing the model to reason over the provided lexical and syntactic information before producing the final text,

$$
y \sim \operatorname { L L M } ( P ) .
$$

To generate N synthetic texts, we repeat this procedure N times, resampling the topic, few-shot exemplars, and candidate words for each generation.

Hyperparameters. Our final configuration uses $k = 3 0 0$ word candidates, $s = 3$ exemplars, a rank boundary of $b = 5 0 0 .$ , a composition of $c = 0 . 1 0$ and a topic fraction of $\rho ~ = ~ 0 . 2 5$ . We selected these values through preliminary experiments in which we generated 5-word sentences and paragraphs under different settings (Appendix C) and asked annotators to rate the fluency of the generated texts.

## 4 Experimental Setup

## 4.1 Configurations

We conduct the experiment with 6 languages, 3 models, 2 thinking modes, 3 prompting strategies, and 5 text lengths, which results in a total of 540 combinations of configurations. We generate 100 outputs for each combination, yielding a total of 54,000 outputs. The configurations are as follows.

Languages. We experiment with six Southeast Asian languages spanning different levels of resource availability: Indonesian and Vietnamese as medium-resource languages, Javanese and Minangkabau as low-resource languages, and Acehnese and Hitu as extremely low-resource languages.

Models and Thinking Modes. We evaluate three LLMs: Gemini 3.5 Flash (Google, 2026), Claude Haiku 4.5 (Anthropic, 2025), and GPT-5.4 Mini (OpenAI, 2026). Each model is evaluated with reasoning enabled and disabled, resulting in six configurations. We initially also considered open-weight models, including Llama 3.3 70B (Grattafiori et al., 2024; Meta AI, 2024) and Qwen2.5 72B (Yang et al., 2024). However, preliminary evaluation showed substantially lower fluency, particularly for low-resource languages. Because our primary fluency evaluation requires native-speaker judgments, we exclude these models from the full evaluation and focus our annotation budget on the stronger model configurations.

Prompting Strategies. We compare three prompting strategies that differ in the amount of information provided to the LLM. The first two serve as baselines, while the third is the strategy used in our method:

1. Topic-Conditioned Zero-Shot Prompt (Zero-shot): This prompt only instructs the model to generate text in the target language about a given topic (see the full prompt in Appendix D).

2. Few-Shot Exemplars Only Prompt (Fewshot): This prompt adds human-written exemplars of the target length to the instruction (see the full prompt in Appendix E).

3. Few-Shot Exemplars and Word Candidates Prompt (Ours): This prompt adds both human-written exemplars of the target length and word candidates sampled from the dictionary to the instruction, as discussed in Section 3 (see the full prompt in Appendix F).

Text Lengths. We generate texts at five sequence lengths: sentences of 5, 10, 15, and 20 words, and paragraphs consisting of 3 sentences and 25 words. This setup lets us test whether the LLM can generate text across a range of lengths.

## 4.2 Evaluation Metrics.

We evaluate the generated outputs along three dimensions: fluency, off-target rate, and lexical diversity. We compute each score at the level of a configuration.

Fluency. We conduct a human evaluation to assess the fluency of the generated texts. Due to annotation costs, we evaluate a subset of outputs covering three models with only thinking mode enabled, three prompting strategies, and three text lengths (5-word sentences, 15-word sentences, and paragraphs), with 30 texts per configuration, resulting in 810 texts per language. Each language is annotated by two annotators. They see only the generated texts, presented in a different random order for each annotator, and complete the annotation independently. They rate each text on a 1–5 scale following our annotation guidelines (see Appendix G), where a higher score indicates a more fluent and natural text. We measure inter-annotator agreement using Pearson and Spearman correlations and observe moderate agreement between the two annotators (see Table 1). The final fluency score for each text is the average of the two annotators’ ratings.

<table><tr><td>Language</td><td>Pearson</td><td>Spearman</td></tr><tr><td>Indonesian</td><td>0.564</td><td>0.243</td></tr><tr><td>Vietnamese</td><td>0.554</td><td>0.516</td></tr><tr><td>Javanese</td><td>0.365</td><td>0.331</td></tr><tr><td>Minangkabau</td><td>0.747</td><td>0.736</td></tr><tr><td>Acehnese</td><td>0.774</td><td>0.771</td></tr><tr><td>Hitu</td><td>0.466</td><td>0.494</td></tr></table>

Table 1: Inter-annotator Pearson and Spearman correlations of human fluency ratings between the two annotators, by language.

Off-Target Rate. We measure off-target rate as the proportion of words in a generated text that do not appear in the target-language dictionary. This metric evaluates a model’s tendency to produce text outside the desired language, reflecting language confusion (Marchisio et al., 2024) or complete failure to produce that language. Scores range from 0 to 1, where 1 means that none of the generated text belongs to the target language, so higher values indicate greater language confusion. We score each output independently and average the scores over each configuration.

Lexical Diversity. We measure lexical diversity with a variant of distinct-n that counts only ngrams composed of target-language words. Let W be the set of target-language words in the dictionary D, and let Y be the set of texts generated under one configuration. Each text $y \in \mathcal { V }$ is a sequence of words $( u _ { 1 } , \ldots , u _ { L } )$ , and an n-gram is a sequence of n consecutive words $( u _ { i } , \dotsc , u _ { i + n - 1 } )$ within a single text. We call an n-gram valid if all of its words belong to W. Let $G _ { n } ( \mathcal { V } )$ be the multiset of all n-grams in Y, and let $U _ { n } ( { \mathcal { Y } } ) \subseteq G _ { n } ( { \mathcal { Y } } )$ be the set of distinct valid n-grams:

$$
U _ { n } ( \mathcal { V } ) = \{ g \in G _ { n } ( \mathcal { V } ) : g _ { 1 } , \ldots , g _ { n } \in \mathcal { W } \} ,\tag{5}
$$

Lexical diversity is the ratio of distinct valid n-grams to all n-grams, averaged over n-gram lengths 1 to $n _ { \mathrm { m a x } }$

$$
\mathrm { L D } ( \mathcal { Y } ) = \frac { 1 } { n _ { \mathrm { m a x } } } \sum _ { n = 1 } ^ { n _ { \mathrm { m a x } } } \frac { | U _ { n } ( \mathcal { Y } ) | } { | G _ { n } ( \mathcal { Y } ) | } ,\tag{6}
$$

where we set $n _ { \mathrm { m a x } } = 5$ . The score lies in [0, 1] and decreases with both repeated n-grams and offtarget words, so higher values indicate a more diverse target-language vocabulary. We compute this score for each configuration.

<table><tr><td rowspan="2">Language</td><td rowspan="2">Prompt</td><td colspan="3">Fluency ↑</td><td rowspan="2">Off-Target Rate↓</td><td colspan="4">Lexical Diversity ↑</td><td rowspan="2"></td></tr><tr><td>5</td><td>15</td><td>Para.</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td colspan="10">Medium-Resource Languages</td></tr><tr><td>Indonesian</td><td>Zero-shot</td><td>4.989</td><td>4.950</td><td>4.761</td><td>0.028</td><td>0.542</td><td>0.573</td><td>0.570</td><td>0.588</td><td>0.546</td></tr><tr><td></td><td>Few-shot</td><td>4.994</td><td>4.994</td><td>4.994</td><td>0.018</td><td>0.430</td><td>0.520</td><td>0.568</td><td>0.608</td><td>0.548</td></tr><tr><td></td><td>Ours</td><td>4.972</td><td>4.911</td><td>4.811</td><td>0.014</td><td>0.868</td><td>0.859</td><td>0.834</td><td>0.817</td><td>0.721</td></tr><tr><td>Vietnamese</td><td>Zero-shot</td><td>4.667</td><td>4.744</td><td>4.722</td><td>0.015</td><td>0.459</td><td>0.546</td><td>0.551</td><td>0.548</td><td>0.508</td></tr><tr><td></td><td>Few-shot</td><td>4.856</td><td>4.828</td><td>4.522</td><td>0.004</td><td>0.457</td><td>0.512</td><td>0.523</td><td>0.545</td><td>0.448</td></tr><tr><td>Ours</td><td></td><td>4.539</td><td>4.017</td><td>3.544</td><td>0.001</td><td>0.876</td><td>0.845</td><td>0.821</td><td>0.800</td><td>0.712</td></tr><tr><td colspan="10">Low-Resource Languages</td></tr><tr><td>Javanese</td><td>Zero-shot</td><td>3.956</td><td>4.239</td><td>4.394</td><td>0.252</td><td>0.266</td><td>0.282</td><td>0.263</td><td>0.273</td><td>0.242</td></tr><tr><td></td><td>Few-shot</td><td>4.267</td><td>4.128</td><td>4.228</td><td>0.200</td><td>0.296</td><td>0.290</td><td>0.311</td><td>0.304</td><td>0.244</td></tr><tr><td></td><td>Ours</td><td>3.744</td><td>3.728</td><td>3.811</td><td>0.090</td><td>0.733</td><td>0.632</td><td>0.609</td><td>0.604</td><td>0.470</td></tr><tr><td>Minangkabau</td><td>Zero-shot</td><td>3.500</td><td>3.561</td><td>3.417</td><td>0.370</td><td>0.234</td><td>0.218</td><td>0.206</td><td>0.208</td><td>0.182</td></tr><tr><td></td><td>Few-shot</td><td>3.894</td><td>3.789</td><td>3.578</td><td>0.221</td><td>0.344</td><td>0.334</td><td>0.315</td><td>0.293</td><td>0.256</td></tr><tr><td></td><td>Ours</td><td>4.317</td><td>4.022</td><td>3.922</td><td>0.064</td><td>0.799</td><td>0.748</td><td>0.674</td><td>0.647</td><td>0.518</td></tr><tr><td colspan="10">Extremely Low-Resource Languages</td></tr><tr><td>Acehnese</td><td>Zero-shot</td><td>3.072</td><td>2.850</td><td>2.633</td><td>0.413</td><td>0.200</td><td>0.189</td><td>0.171</td><td>0.156</td><td>0.136</td></tr><tr><td></td><td>Few-shot</td><td>3.639</td><td>3.239</td><td>2.739</td><td>0.315</td><td>0.234</td><td>0.230</td><td>0.213</td><td>0.198</td><td>0.192</td></tr><tr><td></td><td>Ours</td><td>4.339</td><td>3.889</td><td>3.444</td><td>0.064</td><td>0.702</td><td>0.660</td><td>0.644</td><td>0.614</td><td>0.505</td></tr><tr><td>Hitu</td><td>Zero-shot</td><td>1.789</td><td>1.661</td><td>1.611</td><td>0.836</td><td>0.025</td><td>0.017</td><td>0.016</td><td>0.015</td><td>0.013</td></tr><tr><td></td><td>Few-shot</td><td>2.283</td><td>2.328</td><td>2.117</td><td>0.609</td><td>0.078</td><td>0.058</td><td>0.063</td><td>0.082</td><td>0.042</td></tr><tr><td></td><td>Ours</td><td>3.394</td><td>2.867</td><td>2.828</td><td>0.060</td><td>0.747</td><td>0.670</td><td>0.639</td><td>0.615</td><td>0.509</td></tr></table>

Table 2: Fluency, off-target rate, and lexical diversity per language and prompt. Each value is the average over models and thinking mode (for fluency, only models with thinking mode enabled). Best per language in bold.

## 5 Results

Table 2 reports the results for each language and prompt, averaged across all evaluated models. Results for each individual model and thinking mode are reported in Appendix J (Table 8). We organize the discussion around the evaluation metrics described in Section 4.2.

Fluency. Our prompt achieves the highest fluency for Minangkabau, Acehnese, and Hitu at every text length, exceeding the best baseline by an average of 0.33, 0.69, and 0.79 points, respectively. The margin grows as resource availability decreases and is largest for the extremely low-resource languages, where baseline fluency drops sharply (e.g., 1.61– 1.79 for zero-shot in Hitu). For Indonesian, all prompts score close to the maximum, with few-shot marginally ahead. For Vietnamese and Javanese, however, our prompt trails both baselines, particularly for Vietnamese paragraphs (3.54 vs. 4.72 for zero-shot). We hypothesize that this stems from the quality of the underlying dictionaries: inaccurate or poorly matched entries are passed directly to the model, whereas the baselines rely only on its own knowledge. We examine this hypothesis in Section 6. Finally, fluency generally decreases as

texts get longer.

Off-Target Rate. Our prompt achieves the lowest off-target rate in all six languages. For both baselines, the rate rises steadily as resource availability decreases: with zero-shot prompting, it grows from 0.015–0.028 for the medium-resource languages to 0.836 for Hitu, where most generated words are not in the target language. Our prompt breaks this pattern, keeping the rate at or below 0.090 for every language. The reductions are therefore largest for the extremely low-resource languages, from 0.609 with few-shot prompting to 0.060 for Hitu and from 0.315 to 0.064 for Acehnese. For Indonesian and Vietnamese, all prompts produce very few off-target words, with differences of at most 0.014.

Lexical Diversity. Our prompt yields the highest lexical diversity for every language and text length. The gains hold even for the medium-resource languages, where off-target words are rare: for Indonesian, our prompt scores 0.72–0.87 against 0.43– 0.61 for the baselines, indicating that the word candidates broaden the vocabulary rather than only reducing off-target words. The gap widens as resource availability decreases. For Hitu, the baselines score below 0.09 at every length because their outputs are largely off-target and repetitive, whereas our prompt scores 0.51–0.75, an average improvement of 0.57 over few-shot prompting. For our prompt, lexical diversity decreases steadily with text length in every language and is lowest for paragraphs, partly because longer texts inevitably repeat common words.

<table><tr><td>Language</td><td>Spearman</td><td>MAE</td><td>Bias</td></tr><tr><td>Indonesian</td><td>0.37</td><td>0.12</td><td>-0.04</td></tr><tr><td>Vietnamese</td><td>0.39</td><td>0.44</td><td>+0.40</td></tr><tr><td>Javanese</td><td>0.34</td><td>0.87</td><td>+0.44</td></tr><tr><td>Minangkabau</td><td>0.47</td><td>0.91</td><td>+0.60</td></tr><tr><td>Acehnese</td><td>0.69</td><td>0.86</td><td>+0.58</td></tr><tr><td>Hitu</td><td>0.15</td><td>1.70</td><td>+1.19</td></tr></table>

Table 3: Reliability of the LLM judge (Gemini 2.5 Flash) on fluency evaluation, measured against human ratings.

## 6 Analysis

Dictionary Qualities Hinder Fluency in Vietnamese and Javanese. Our method scores below the baselines in fluency for Vietnamese and Javanese. To investigate these cases, we asked the annotators to examine the generated texts and the vocabulary entries provided to the model. Their feedback suggests that dictionary quality plays an important role. For Vietnamese, some dictionary entries were valid but uncommon in everyday usage, while more familiar alternatives were missing. As a result, semantically correct generations could still sound unnatural to native speakers. For Javanese, the dictionary primarily represents the Yogyakarta variety, whereas our annotator speaks Javanese with Arekan dialect. This mismatch led to unfamiliar vocabulary appearing in the generated text, reducing perceived fluency despite the words being valid dictionary entries. These observations highlight an important practical consideration for vocabulary-guided generation: the dictionary should be validated for both lexical coverage and alignment with the target language variety before large-scale generation. In particular, commonly used words should be adequately represented, and the lexical resource should match the variety spoken by the intended users.

Reliability of the LLM-as-a-Judge for Fluency Evaluation. We assess the reliability of the LLMas-a-judge by comparing the fluency scores it assigns against the human fluency scores collected in our previous human evaluation (Section 4.2) using the mean of the two annotators’ scores across six languages. We use Gemini 2.5 Flash (Google, 2025) as the judge. For each language, let $h _ { i }$ denote the mean of the human fluency scores and $j _ { i }$ the judge’s score for the i-th output, over $N$ outputs. We report Spearman’s rank correlation $\rho$ between $h _ { i }$ and $j _ { i }$ to measure how well the judge preserves the human ranking of outputs, Mean Absolute Error (MAE) to measure how far its scores deviate from human scores, and mean signed bias to measure whether it over- or underestimates fluency.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Thinking</td><td rowspan="2">Off-Target Rate↓</td><td rowspan="2">Lexical Diversity ↑</td></tr><tr><td></td></tr><tr><td>GPT</td><td>Off</td><td>0.347</td><td>0.311</td></tr><tr><td rowspan="2">Gemini</td><td>On</td><td>0.322</td><td>0.341</td></tr><tr><td>Off</td><td>0.294</td><td>0.336</td></tr><tr><td rowspan="2">Claude</td><td>On</td><td>0.233</td><td>0.371</td></tr><tr><td>Off</td><td>0.378</td><td>0.300</td></tr><tr><td></td><td>On</td><td>0.394</td><td>0.290</td></tr></table>

Table 4: Results per model and thinking mode, averaged over three languages (Hitu, Minangkabau, and Acehnese) and three prompts. Lexical diversity is averaged over the 5-, 10-, 15-, 20-word and paragraph settings. The best score between the thinking modes is shown in bold.

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| j _ { i } - h _ { i } \right|
$$

$$
{ \mathrm { B i a s } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \left( j _ { i } - h _ { i } \right)
$$

The judge’s reliability varies across languages, as shown in Table 3. Spearman’s correlation is moderate for most languages, strongest for Acehnese (0.69), and weakest for Hitu (0.15), suggesting that the judge’s rankings of Hitu texts align poorly with human judgments of fluency. MAE and bias show a clearer pattern. For Indonesian, the judge’s scores show the closest agreement with human ratings, with the lowest MAE (0.12) and a near-zero bias (-0.04). For the remaining languages, the MAE is generally higher, and the bias is consistently positive, indicating that the judge tends to assign higher scores than the human. This overestimation is particularly pronounced for Minangkabau (+0.60) and Acehnese (+0.58), and greatest for Hitu, which also has the highest MAE (+1.19).

![](images/c664bf1582fb5e2b22d8a38cf3eab548c4dad8719cadce8aade2fdb686456ed1.jpg)  
Figure 2: Average percentage of unique valid n-grams as the number of generated Minangkabau sentences increases.

Overall, the results show that the judge’s agreement with human annotators varies across languages. While its fluency rankings align relatively well with human judgments for Acehnese, it tends to assign higher scores than human annotators and has larger scoring errors for several lower-resource languages. Therefore, LLM-as-a-judge should be used with caution for fluency evaluation in these languages and should not be considered a substitute for human evaluation.

Thinking vs. Non-thinking Mode. Table 4 compares the thinking mode between enabled and disabled mode for the three LRLs (Hitu, Minangkabau, and Acehnese). Enabling thinking tends to improve generation quality, but the effect depends on the model. Gemini 3.5 Flash benefits most, reducing the off-target rate by 0.06 and increasing lexical diversity by 0.03. GPT-5.4 Mini shows a smaller gain, with the off-target rate decreasing by 0.02 and lexical diversity increasing by 0.03. Claude Haiku 4.5 is the exception, as thinking brings no benefit. Overall, thinking helps in most cases, but the benefit is model-dependent.

Robustness Under Increasing Generation Volume. We investigate whether generating larger volumes of text leads to reduced lexical diversity and whether our prompt mitigates this effect compared with the baselines. We run this stress test experiment in Minangkabau and Hitu, generating up to 1,000 five-word sentences for each prompting strategy using Gemini 3.5 Flash with thinking mode enabled. We compute the percentage of unique valid n-grams among those observed in the first N sentences, where $N = 1 0 0 , 2 0 0 , \ldots , 1 0 0 0$

![](images/514cdb5e079a106958a13870196ad08bb26a1d05e116f583d52c8b23f95547ae.jpg)  
Figure 3: Average percentage of unique valid n-grams as the number of generated Hitu sentences increases.

Figures 2 and 3 show that this percentage decreases as more sentences are generated across all prompting strategies, which is expected as repeated n-grams accumulate. In Minangkabau, the zeroshot and few-shot scores are 0.32 and 0.36 at 100 sentences and drop to between 0.14 and 0.15 at 1,000 sentences. The same pattern holds in Hitu, where the zero-shot and few-shot scores fall from 0.16 and 0.05 to 0.09 and 0.04, respectively. However, our prompt consistently maintains a higher percentage than the baselines in both languages, suggesting that it better preserves lexical diversity as generation volume increases.

## 7 Conclusion

We introduced a training-free framework that leverages human-curated vocabularies, human-written exemplars, and LLM reasoning to generate synthetic text in low-resource languages. Across six languages, our framework produces more lexically diverse text with fewer off-target words than the baselines, while improving fluency particularly for the extremely low-resource Acehnese and Hitu. These results show the potential of human-curated vocabularies for bootstrapping synthetic text corpora where existing text resources are limited.

Our analysis also highlights important practical considerations. Dictionary quality and alignment with the target language variety can substantially affect generation quality, making vocabulary validation important before large-scale generation. The framework also requires additional input tokens for word candidates and exemplars, creating a trade-off between generation quality and cost. Future work should extend the framework to more low-resource languages and investigate whether the resulting synthetic corpora improve LLM pretraining and post-training.

## Limitations

Our experiments cover six Southeast Asian languages, five of which are Austronesian and all written in Latin script. Five are spoken in Indonesia and share considerable lexical material with Indonesian, a setting that is likely to favor positive transfer. We therefore cannot claim that the framework generalizes to non-Latin scripts or to agglutinative or polysynthetic morphology. We further describe Hitu as unseen during pretraining, but the training corpora of the three commercial models we use are not public, so this characterization is inferred from model behavior rather than established, as illustrated in Appendix H.

As our experiments are extensive, generating 54,000 texts made broad model coverage too costly, so we limit our study to three proprietary models accessed through commercial APIs, all from the economy tier of their respective families. Our findings are therefore restricted to these models, which are the latest versions of these model families at the time of our experiments. It remains unclear whether the gains hold for larger or more recent models, which may already encode more of these languages.

Furthermore, we evaluate the fluency, off-target rate, and lexical diversity of the generated text, but we have not trained or fine-tuned any model on the resulting data. With the quality gains in fluency and lexical diversity reported in Table 2, we see this as a promising direction for future work: testing whether our method benefits downstream models, particularly for LRLs.

## Ethical Considerations

This work involved native speakers of the six target languages, who wrote the few-shot exemplars (Appendix B) and rated the fluency of the generated texts (Appendix G). Before agreeing to participate, annotators were informed of the purpose of the study, the expected workload, and the intended research use of their contributions. They were paid above the regional minimum wage in Indonesia and Vietnam and were free to withdraw at any point. No personal or sensitive information was collected, and none appears in our released materials. The Javanese, Minangkabau, Hitu, and Acehnese dictionaries are published by the Indonesian Ministry of Primary and Secondary Education, and the Indonesian and Vietnamese resources are distributed under Apache-2.0 and MIT licences; we use all resources for non-commercial academic research only and cite them in full (Appendix A).

## Acknowledgments

This research was conducted through the apprenticeship program organized by the SEACrowd community. We are grateful to SEACrowd for facilitating this initiative and supporting emerging researchers in their early research journey by providing a platform for research collaboration and learning.

## References

Tatiana Anikina, Jan Cegin, Jakub Simko, and Simon Ostermann. 2025. A rigorous evaluation of LLM data generation strategies for low-resource languages. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 8282–8303, Suzhou, China. Association for Computational Linguistics.

Anthropic. 2025. System card: Claude Haiku 4.5. Technical report, Anthropic. Accessed 2026-10-13.

Niyati Bafna, Tianjian Li, Kenton Murray, David R. Mortensen, David Yarowsky, Hale Sirin, and Daniel Khashabi. 2025. The translation barrier hypothesis: Multilingual generation with large language models suffers from implicit translation failure. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 1541–1568, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Jan Cegin, Branislav Pecher, Jakub Simko, Ivan Srba, Maria Bielikova, and Peter Brusilovsky. 2025. Use random selection for now: Investigation of few-shot selection strategies in LLM-based text augmentation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 5533–5550, Suzhou, China. Association for Computational Linguistics.

Meet Doshi, Raj Dabre, and Pushpak Bhattacharyya. 2024. Pretraining language models using translationese. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 5843–5862, Miami, Florida, USA. Association for Computational Linguistics.

Steven Y. Feng, Varun Gangal, Jason Wei, Sarath Chandar, Soroush Vosoughi, Teruko Mitamura, and Eduard Hovy. 2021. A survey of data augmentation approaches for NLP. In Findings of the Association

for Computational Linguistics: ACL-IJCNLP 2021, pages 968–988, Online. Association for Computational Linguistics.

Google. 2025. Gemini 2.5 Flash. https: //ai.google.dev/gemini-api/docs/models/ gemini-2.5-flash. Released 17 June 2025. Accessed 2026-10-13.

Google. 2026. Gemini 3.5 Flash. https: //ai.google.dev/gemini-api/docs/models/ gemini-3.5-flash. Released 19 May 2026. Accessed 2026-10-13.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Yanzhu Guo, Guokan Shang, Michalis Vazirgiannis, and Chloé Clavel. 2024. The curious decline of linguistic diversity: Training language models on synthetic text. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 3589–3604, Mexico City, Mexico. Association for Computational Linguistics.

Gauri Kambhatla, Chantal Shaib, and Venkata S Govindarajan. 2025. Measuring lexical diversity of synthetic data generated through fine-grained persona prompting. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 21024– 21033, Suzhou, China. Association for Computational Linguistics.

Akbar Karimi, Leonardo Rossi, and Andrea Prati. 2021. AEDA: An easier data augmentation technique for text classification. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2748–2754, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yue Li, Zhixue Zhao, and Carolina Scarton. 2025. It’s all about in-context learning! teaching extremely lowresource languages to LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 29544–29559, Suzhou, China. Association for Computational Linguistics.

Kelly Marchisio, Wei-Yin Ko, Alexandre Bérard, Théo Dehaze, and Sebastian Ruder. 2024. Understanding and mitigating language confusion in LLMs. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 6653– 6677, Miami, Florida, USA. Association for Computational Linguistics.

Meta AI. 2024. Llama 3.3 70B instruct model card. https://huggingface.co/meta-llama/ Llama-3.3-70B-Instruct. Accessed 2026-10-13.

Amani Namboori, Shivam Sadashiv Mangale, Andy Rosenbaum, and Saleh Soltan. 2023. GeMQuAD : Generating multilingual question answering datasets from large language models using few shot learning. In NeurIPS 2023 Workshop on Synthetic Data Generation with Generative AI.

OpenAI. 2026. Introducing GPT-5.4 mini and nano. https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/. Released 17 March 2026. Accessed 2026-10-13.

Wei Peng, Chongxuan Huang, Tianhao Li, Yun Chen, and Qun Liu. 2020. Dictionary-based data augmentation for cross-domain neural machine translation. Preprint, arXiv:2004.02577.

Salsabila Zahirah Pranida, Rifo Ahmad Genadi, and Fajri Koto. 2025. Culturally-nuanced story generation for reasoning in low-resource languages: The case of Javanese and Sundanese. In Proceedings ofthe 5th Workshop on Multilingual Representation Learning (MRL 2025), pages 369–384, Suzhuo, China. Association for Computational Linguistics.

Mathias Stenlund, Annika Simonsen, Lars Bungum, Jan Ebert, Jiangtao Wang, Oleg Filatov, Hemanadhan Myneni, Morris Riedel, and Hafsteinn Einarsson. 2026. Synthetic instruction generation for lowresource Nordic languages: Viability and limitations in LLM instruction-tuning. In Proceedings ofthe Fifteenth Language Resources and Evaluation Conference, pages 10688–10698, Palma de Mallorca, Spain. ELRA Language Resource Association.

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. 2024. A benchmark for learning to translate a new language from one grammar book. In International Conference on Learning Representations, volume 2024, pages 18955–18985.

Xinyi Wang, Sebastian Ruder, and Graham Neubig. 2022. Expanding pretrained models to thousands more languages via lexicon-based adaptation. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 863–877, Dublin, Ireland. Association for Computational Linguistics.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, Toronto, Canada. Association for Computational Linguistics.

Jason Wei and Kai Zou. 2019. EDA: Easy data augmentation techniques for boosting performance on text classification tasks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6382–6388, Hong Kong, China. Association for Computational Linguistics.

An Yang, Baosong Yang, Beichen Zhang, et al. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Zheng Xin Yong, Cristina Menghini, and Stephen Bach. 2024. LexC-gen: Generating data for extremely lowresource languages with large language models and bilingual lexicons. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 13990–14009, Miami, Florida, USA. Association for Computational Linguistics.

Chen Zhang, Xiao Liu, Jiuheng Lin, and Yansong Feng. 2024. Teaching large language models an unseen language on the fly. In Findings ofthe Association for Computational Linguistics: ACL 2024, Bangkok, Thailand. Association for Computational Linguistics.

Tongyao Zhu, Qian Liu, Chang Ma, Jinghan Zhang, Longxu Dou, Junxian He, and Shiqi Chen. 2026. How can synthetic data improve multilingual language model pretraining? a data quality perspective. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21943–21956, San Diego, California, United States. Association for Computational Linguistics.

## A Dictionary Resources

The available resources vary across the languages used in our experiments. For Javanese<sup>5</sup> and Minangkabau<sup>6</sup>, we collect dictionary entries through web scraping. Some entries in the Javanese dictionary lack information such as POS and example sentences. We use Gemini 3.5 Flash to fill in these missing fields based on the word and definition. For Hitu<sup>7</sup> and Acehnese<sup>8</sup>, the dictionary resources are available as PDF documents. We therefore use Gemini 3.5 Flash to extract the content from the scanned dictionary pages and convert it into a structured dataset. For Indonesian<sup>9</sup> and Vietnamese<sup>10</sup>, we use an existing dictionary dataset available on Hugging Face. We further process the entries to obtain a consistent format across languages. For non-English definitions, we use Google Translate<sup>11</sup> to translate them into English. We show the collected dictionary size in Table 5.

Copyright and licensing. All resources are used solely for non-commercial academic research. The Javanese, Minangkabau, Hitu, and Acehnese dictionaries are published by the Indonesian Ministry of Primary and Secondary Education (Kemendikdasmen), and we cite the sources in full as permitted under Article 44 of the Indonesian Copyright Law (UU No. 28 Tahun 2014).<sup>12</sup> The Indonesian and Vietnamese datasets are obtained from Hugging Face, where their licenses are stated as Apache-2.0 and MIT, respectively.

<table><tr><td>Language</td><td>Number of Words</td></tr><tr><td>Indonesian</td><td>71,017</td></tr><tr><td>Javanese</td><td>65,568</td></tr><tr><td>Vietnamese</td><td>33,968</td></tr><tr><td>Minangkabau</td><td>33,149</td></tr><tr><td>Acehnese</td><td>10,638</td></tr><tr><td>Hitu</td><td>1,347</td></tr></table>

Table 5: Number of words in the dictionary of each language.

## B Few-Shot Exemplar Creation Procedure

We assigned one native-speaker annotator to each target language. Each annotator is a native speaker of or fluent in the target local language and has lived in the region where the language is spoken for a period of time.

The exemplar creation process began with the development of annotation guidelines. Before commencing the task, annotators were asked to read the guidelines carefully and were encouraged to raise any questions or seek clarification about aspects they found unclear.

Each annotator was provided with a separate Google Sheets document to record the exemplars they created, along with the corresponding dictionary dataset to facilitate the selection of candidate words. The dictionary entries provided lexical information, including the word, part of speech, English definition, and example sentence. Annotators were instructed to use words from the provided dictionary dataset whenever possible when constructing the exemplars. A Jupyter Notebook was also provided to assist annotators in searching relevant dictionary entries. Figure 4 presents the guidelines.

## Exemplar Creation Guidelines

The annotation task consists of creating 40 sentences and 5 paragraphs in the target language. The sentences should be distributed as follows:

```typescript
word | pos | definition | example
```

• 10 sentences of approximately 5 words each;

• 10 sentences of approximately 10 words each;

• 10 sentences of approximately 15 words each;

• 10 sentences of approximately 20 words each; and

• 5 paragraphs, with each paragraph containing at least 3 sentences and at least 25 words.

Small variations in sentence length are acceptable as long as the sentence remains natural. For example, a target of 5 words corresponds approximately to 4–6 words, while a target of 10 words corresponds approximately to 9–11 words.

For each exemplar, annotators should fill the following fields:

• sentence: the sentence or paragraph written in the target language;

• translation: the English translation of the sentence or paragraph; and

• words\_candidates: the dictionary entries corresponding to the words used to construct the sentence or paragraph.

For the words\_candidates field, annotators should identify the dictionary entry corresponding to each word used in the sentence whenever possible. Dictionary entries are provided in the following format:

where word is the word in the target language, pos is its part of speech, definition is its English definition, and example is an example sentence when available.

Annotators should use words from the provided dictionary dataset whenever possible. If an additional word is needed to produce a natural and grammatically correct sentence but is not available in the dictionary dataset, the annotator may introduce it as a new word candidate. The new word should also be recorded using the same dictionary entry format.

Figure 4: Guidelines for creating few-shot exemplars.

## C Hyperparameter Settings

We selected the final hyperparameters through preliminary experiments. For each candidate configuration, we generated 15 five-word sentences and paragraphs, and annotators rated the fluency of the generated texts. Table 6 lists the hyperparameters and the values we searched over.

<table><tr><td>Param.</td><td>Description</td><td>Values</td></tr><tr><td>k</td><td>Number of word candidates</td><td>[100, 300, 500]</td></tr><tr><td>n</td><td>Number of few-shot examples</td><td>[3, 5, 7, 10]</td></tr><tr><td>b</td><td>Rank boundary</td><td>[100, 500, 1000]</td></tr><tr><td>C</td><td>Composition</td><td>[10%, 20%, 50%]</td></tr><tr><td>p</td><td>Top-p similarity fraction</td><td>[25%, 50%]</td></tr></table>

Table 6: Hyperparameter settings.

## D Topic-Conditioned Zero-Shot Prompt

Figure 5 presents the prompt template used for topic-related generation. The [TARGET\_LANGUAGE], [OUTPUT\_TYPE], [TOPIC], and [TARGET\_SENTENCE\_LENGTH] are filled by the target language (Hitu, Indonesian, Javanese, Minangkabau, Vietnamese, or Acehnese), the output type (SENTENCE or PARAGRAPH), a single topic, and the target sentence length (5, 10, 15, or 20 words), respectively. The topic is one of the following 20 topics: food, family, holiday, travel, health, fitness, education, technology, music, movies, books, fashion, nature, pets, business, art, science, gaming, home, and garden.

Generate a [OUTPUT\_TYPE] in the   
[TARGET\_LANGUAGE] language about   
[TOPIC].   
IF OUTPUT\_TYPE == SENTENCE:   
Ensure that the sentence is around   
[TARGET\_SENTENCE\_LENGTH] words long.   
IF OUTPUT\_TYPE == PARAGRAPH:   
Ensure that the paragraph contains at least 3   
sentences and at least 25 words.   
Return only the [OUTPUT\_TYPE].  
Figure 5: Template for topic-conditioned zero-shot prompt.

## E Few-Shot Exemplars Only Prompt

Figure 6 presents the prompt template used for generation with few-shot exemplars only, without word candidates. The [TARGET\_LANGUAGE], [OUTPUT\_TYPE], and [TARGET\_SENTENCE\_LENGTH] are filled by the target language, the output type (SENTENCE or PARAGRAPH), and the target sentence length (5, 10, 15, or 20 words), respectively. The few-shot examples ({FEW\_SHOT\_EXAMPLE\_1} to {FEW\_SHOT\_EXAMPLE\_N}) are N sample outputs in the target language, which serve only as guidance for style, grammar, and fluency.

You are a linguistic expert in   
[TARGET\_LANGUAGE].   
TASK: Create one coherent [OUTPUT\_TYPE]   
in [TARGET\_LANGUAGE].   
===   
RULES (MUST FOLLOW):   
===   
1. FEW-SHOT GUIDANCE:   
- Use the few-shot examples only as guidance   
for writing style, grammar, fluency, and natu  
ral sentence structure.   
- Do not reuse the main entities, events, or   
semantic content from any few-shot example.   
- Do not create output that is a paraphrase, ex  
tension, recombination, or slight modification   
of a few-shot example.   
2. PUNCTUATION & CASING:   
- You ARE allowed to use standard punctua  
tion.   
- You ARE allowed to capitalize the first letter   
of a word.   
3. LENGTH & COVERAGE:   
IF OUTPUT\_TYPE == SENTENCE:   
- Output exactly one sentence.   
The sentence should be around   
[TARGET\_SENTENCE\_LENGTH] words.   
IF OUTPUT\_TYPE == PARAGRAPH:   
- Output a paragraph containing at least 3   
sentences.   
- The paragraph must contain at least 25   
words.   
4. QUALITY:   
- The output must be grammatically correct in   
[TARGET\_LANGUAGE].

![](images/f2d565eb8f91784f068de05a59e8477180fb4e41bb91eae1e625708b2485f680.jpg)  
Figure 6: Template for few-shot exemplars only prompt.

## F Few-Shot Exemplars and Word Candidates Prompt

Figure 7 presents the prompt template used for generation with few-shot exemplars and word candidates. The [TARGET\_LANGUAGE], [OUTPUT\_TYPE], and [TARGET\_SENTENCE\_LENGTH] are filled by the target language (Hitu, Indonesian, Javanese, Minangkabau, Vietnamese, or Acehnese), the output type (SENTENCE or PARAGRAPH), and the target sentence length (5, 10, 15, or 20 words), respectively. The few-shot examples consist of N exemplars. Each exemplar i contains a list of word candidates ([WORD\_CANDIDATES\_i]), formatted as <word> | <part-of-speech> | <meaning> | <example\_usage>, followed by its corresponding output in the target language ([FEW\_SHOT\_EXAMPLE\_i]). The final [WORD\_CANDIDATES] are the candidate words, in the same format, for which a new output is to be generated.

You are a linguistic expert in   
[TARGET\_LANGUAGE].   
TASK: Create one coherent [OUTPUT\_TYPE]   
in [TARGET\_LANGUAGE] that sounds natural   
and fluent to a native speaker.   
RULES (MUST FOLLOW):   
1. VOCABULARY:   
- Each entry in the “Word Candidates” list fol-

lows this structure, separated by “ | ”:   
<word> | <part-of-speech> | <meaning> | <ex  
ample\_usage>   
where:

\- <word>: the [TARGET\_LANGUAGE] word to use.

\- <part-of-speech>: the grammatical role of the word (e.g., noun, verb, adjective).

\- <meaning>: the English meaning of the word.

\- <example\_usage>: one or more example sentences in [TARGET\_LANGUAGE] showing how the word is naturally used, for reference only. If multiple examples are given, they are separated by “ \* ”.

\- Use the word candidates list as vocabulary guidance rather than a strict requirement.

\- Try to use the candidate words where they fit naturally in the output.

The sentence should be around [TARGET\_SENTENCE\_LENGTH] words. IF OUTPUT\_TYPE == PARAGRAPH:

\- Output a paragraph containing at least 3 sentences.

\- The paragraph must contain at least 25 words.

3. PUNCTUATION & CASING:

\- You ARE allowed to use standard punctuation (periods, commas, question marks).

\- You ARE allowed to capitalize the first letter of a word to start the output or for proper nouns.

## 4. QUALITY:

\- The output must be grammatically correct in [TARGET\_LANGUAGE].

\- DO NOT prioritize length over logical coherence. Prioritize natural, coherent, and meaningful output.

\- DO NOT use specialized/technical terms unless they fit the context perfectly.

\- Do not reuse the main entities, events, or semantic content from any few-shot example.

\- Do not create output that is a paraphrase, extension, recombination, or slight modification of a few-shot example.

![](images/0c5cdb90fa6875d2c16cc39ff23cdd087b94108e7d903b1ff12237be5a6f54a1.jpg)  
Figure 7: Template for few-shot exemplars and word candidates prompt.

## G Human Evaluation Guidelines

Each annotator received a spreadsheet of the 810 generated texts for their language, with the texts in an output column, together with the instructions shown in Figure 8.

your task is to rate the fluency of each given text on a 1–5 Likert scale.

Scoring Rubric:

1. Very Poor: The text is largely unintelligible due to severe grammatical errors, broken syntax, or nonsensical word order. The intended meaning is difficult or impossible to determine. This score also applies if the text is an automated error response or is written in another language.

2. Poor: The intended meaning is mostly recoverable, but frequent grammatical errors, incorrect word order, or unnatural phrasing make the text difficult to read.

3. Fair: The meaning is clear, but the text contains some grammatical errors or awkward constructions that reduce its naturalness.

4. Good: The text is grammatically correct and easy to understand, with only minor stylistic awkwardness or slightly unnatural wording.

5. Excellent: The text is completely fluent, natural, and reads as if written by a proficient native speaker.

## Examples:

1. Very Poor: “Yesterday market she because apples the quickly and was.”

2. Poor: “She go market yesterday buy apples for she family.”

3. Fair: “She went to market yesterday for buying apples for her family.”

4. Good: “She went to the market yesterday to buy apples for her family.”

5. Excellent: “She went to the market yesterday to buy fresh apples for her family, since they had run out at home.”

## H Output Sample Comparison Between Baseline and Our Pipeline

Table 7 illustrates how our method outperforms the baseline in Hitu using Claude Haiku 4.5. Without the few-shot exemplars and word candidates, the model fails to generate Hitu sentences and paragraphs because of its limited knowledge of the language. With the few-shot exemplars and the word candidates that offer vocabulary options (and without further training), the same model generates more fluent text in this language.

## I Information About AI Usage

We used AI assistants only for polishing the writing and for literature search. We manually read and

Figure 8: Instructions given to human annotators for the fluency evaluation.
<table><tr><td>Language</td><td>Generated Text</td><td>Human Score</td></tr><tr><td colspan="2">5-word sentences</td><td></td></tr><tr><td>Zero-shot</td><td>I don&#x27;t have reliable knowledge of the Hitu language to generate an accurate sentence. Hitu is a small Austronesian language spoken in Indonesia, and ...</td><td>1</td></tr><tr><td>Ours</td><td>Malole ma laka tarus nela.</td><td>5</td></tr><tr><td>Paragraphs Zero-shot</td><td>I appreciate your request, but I must be honest: I don&#x27;t have reliable knowledge of the Hitu language to generate authentic text in it. Hitu is a language</td><td>1</td></tr><tr><td>Ours</td><td>spoken in Indonesia&#x27;s Maluku Islands ... Baba laha ina una hetu ele luma asusu kula hatu pata maroto nale tolua. Sitewa manusia lalutu sataenai nai hahitin luma ma lama ingkuat hele. Ite manumpang kareta le lahan mahain ena supulumai tania kalu</td><td>3</td></tr></table>

Table 7: The sample outputs of Claude Haiku 4.5 in zero-shot and our method at the 5-word sentence and paragraph level.

verified all papers found through AI-assisted search, and we take full responsibility for the content of this paper. All ideas originate from the authors.

## J Detailed Results by Model and Thinking Mode

Table 8 reports fluency, off-target rate, and lexical diversity for each language, model, thinking mode, and prompt.

<table><tr><td></td><td></td><td></td><td></td><td colspan="3">Fluency ↑</td><td>Off-Target Rate ↓</td><td colspan="6">Lexical Diversity ↑</td></tr><tr><td>Language</td><td>Model</td><td>Thinking</td><td>Prompt</td><td>5</td><td></td><td>15</td><td>Para.</td><td></td><td>5</td><td>10</td><td>15</td><td>20</td><td>Para.</td></tr><tr><td colspan="2">Indonesian</td><td>On</td><td>Zero-shot</td><td>4.967</td><td></td><td>4.933</td><td>Medium-Resource Languages 4.400 5.000</td><td>0.036 0.014</td><td>0.577</td><td>0.541</td><td>0.578</td><td>0.606</td><td>0.565 0.569</td></tr><tr><td colspan="2"></td><td>Off</td><td>Few-shot Ours Zero-shot</td><td></td><td>4.983 4.967</td><td>5.000 4.917</td><td>4.700</td><td>0.015 0.029 0.015</td><td>0.533 0.888 0.579</td><td>0.556 0.857 0.543</td><td>0.546 0.824 0.573 0.580</td><td>0.618 0.817 0.715 0.569 0.549 0.612</td></tr><tr><td colspan="2"></td><td>Gemini On</td><td>Few-shot Ours Zero-shot</td><td></td><td>5.000</td><td>5.000</td><td>4.933</td><td>0.015 0.028 0.025</td><td>0.420 0.873 0.542 0.420</td><td>0.465 0.869 0.641</td><td>0.849 0.828 0.629 0.681 0.673 0.650</td><td>0.543 0.728 0.556 0.546</td></tr><tr><td colspan="2"></td><td>Off</td><td>Few-shot Ours Zero-shot Few-shot</td><td>5.000 4.983</td><td>5.000 4.950</td><td>5.000 4.917</td><td></td><td>0.011 0.029 0.016</td><td>0.874 0.861 0.454</td><td>0.656 0.828 0.473 0.429</td><td>0.822 0.491 0.395</td><td>0.724 0.511 0.530</td></tr><tr><td colspan="2">Claude</td><td>On</td><td>Ours Zero-shot Few-shot</td><td>5.000 5.000</td><td>4.917</td><td>4.950</td><td></td><td>0.013 0.023</td><td>0.191 0.869 0.590</td><td>0.240 0.261 0.869 0.847 0.724 0.678</td><td>0.826 0.660 0.697</td><td>0.730 0.569 0.619</td></tr><tr><td colspan="2"></td><td>Off</td><td>Ours Zero-shot</td><td>4.967</td><td>4.867</td><td>4.983</td><td>4.983 4.817</td><td>0.023 0.014 0.025</td><td>0.600 0.845 0.510</td><td>0.686 0.846 0.517</td><td>0.719 0.824 0.789 0.536 0.519</td><td>0.714 0.526</td></tr><tr><td colspan="2">Vietnamese GPT</td><td></td><td>Few-shot Ours Zero-shot</td><td>4.567</td><td>4.683</td><td></td><td>4.683</td><td>0.018 0.016 0.014</td><td>0.416 0.858 0.554</td><td>0.516 0.850 0.605</td><td>0.631 0.676 0.832 0.817 0.578 0.578</td><td>0.481 0.716 0.532</td></tr><tr><td colspan="2"></td><td>On</td><td>Few-shot Ours Zero-shot</td><td>4.850 4.467</td><td>4.817 3.933</td><td></td><td>4.417 3.050</td><td>0.001 0.001 0.014</td><td>0.447 0.906 0.427</td><td>0.507 0.879 0.852</td><td>0.453 0.516 0.833 0.535 0.550</td><td>0.452 0.737 0.535</td></tr><tr><td colspan="2"></td><td>Off</td><td>Few-shot Ours Zero-shot</td><td>4.900</td><td></td><td></td><td></td><td>0.001 0.001 0.017</td><td>0.377 0.900 0.399</td><td>0.514 0.384 0.322 0.855 0.857</td><td>0.435 0.830 0.613</td><td>0.374 0.743 0.493</td></tr><tr><td colspan="2"></td><td>Gemini On</td><td>Few-shot Ours Zero-shot</td><td>4.950 4.817</td><td>4.900 4.933 4.417</td><td></td><td>4.933 4.817 4.133</td><td>0.007 0.002 0.018</td><td>0.399 0.861</td><td>0.639 0.667 0.523 0.600 0.830 0.797</td><td>0.551 0.767</td><td>0.465 0.683 0.441</td></tr><tr><td colspan="2">Claude On</td><td>Off</td><td>Few-shot Ours Zero-shot</td><td>4.533</td><td>4.650</td><td></td><td></td><td>0.005 0.001 0.015</td><td>0.361 0.444 0.430 0.867 0.817</td><td>0.405 0.382 0.488 0.791</td><td>0.439 0.510 0.778 0.621</td><td>0.400 0.690 0.538</td></tr><tr><td colspan="2"></td><td>Off</td><td>Few-shot Ours Zero-shot</td><td>4.767 4.333</td><td>4.733 3.700</td><td></td><td>4.550 4.333 3.450</td><td>0.007 0.002 0.014</td><td>0.558 0.640 0.841 0.457</td><td>0.644 0.654 0.685 0.711 0.828 0.802</td><td>0.701 0.790 0.493 0.487</td><td>0.486 0.711 0.508</td></tr><tr><td colspan="2">Low-Resource Languages</td><td></td><td>Few-shot Ours</td><td></td><td></td><td></td><td></td><td>0.005 0.001</td><td>0.433 0.881</td><td>0.471 0.543 0.860</td><td>0.566 0.557 0.827 0.802</td><td>0.512 0.708</td></tr><tr><td colspan="2">Javanese GPT</td><td>On</td><td>Few-shot</td><td>Zero-shot</td><td>4.200 4.550</td><td>4.417 4.217</td><td>4.483 4.317</td><td>0.207 0.153</td><td>0.286 0.300</td><td>0.341 0.354</td><td>0.308 0.322 0.358 0.353</td><td>0.307 0.288</td></tr><tr><td colspan="2"></td><td>Off</td><td>Ours Zero-shot</td><td></td><td>3.383</td><td>3.683</td><td>3.850</td><td>0.105 0.217 0.141</td><td>0.753 0.314 0.295</td><td>0.616 0.582 0.295</td><td>0.576 0.300</td><td>0.448 0.288</td></tr><tr><td colspan="2"></td><td></td><td>Few-shot Ours</td><td></td><td></td><td></td><td></td><td>0.090</td><td>0.367 0.736</td><td>0.308 0.316 0.642 0.604</td><td>0.343 0.577</td><td>0.304 0.486</td></tr><tr><td colspan="2"></td><td>Gemini On</td><td>Zero-shot</td><td>4.550 Few-shot 4.583</td><td>4.267</td><td>4.150</td><td>4.583 4.483</td><td>0.259 0.201</td><td>0.196 0.279 0.276</td><td>0.277 0.265</td><td>0.286 0.335 0.318</td><td>0.228 0.220</td></tr></table>

Continued on next page

Table 8 – continued from previous page
<table><tr><td colspan="4"></td><td colspan="3">Fluency ↑</td><td rowspan="2">Off-Target Rate ↓</td><td colspan="4">Lexical Diversity ↑</td><td rowspan="2">Para.</td></tr><tr><td>Language</td><td>Model</td><td>Thinking Prompt</td><td>Zero-shot</td><td>5</td><td>15</td><td>Para.</td><td></td><td>5</td><td>10 15</td><td>20</td></tr><tr><td>Minangkabau</td><td>Claude</td><td>On Off</td><td>Few-shot Ours Zero-shot</td><td>3.117 3.667 3.867</td><td>4.150 3.900 3.650</td><td>4.117 3.883 3.683</td><td>0.279 0.258 0.089</td><td>0.322 0.344 0.751</td><td>0.294 0.323 0.624</td><td>0.247 0.333 0.586</td><td>0.262 0.297 0.599</td><td>0.220 0.244 0.436 0.201</td></tr><tr><td rowspan="6"></td><td></td><td></td><td>Few-shot Ours Zero-shot</td><td>3.967</td><td>3.733</td><td>3.467</td><td>0.283 0.251 0.085 0.310</td><td>0.299 0.283 0.701 0.291</td><td>0.235 0.283 0.642 0.240</td><td>0.236 0.263 0.631 0.239</td><td>0.231 0.278 0.614 0.236</td><td>0.215 0.496 0.250</td></tr><tr><td>GPT</td><td>On</td><td>Few-shot Ours</td><td>4.183 4.367</td><td>3.883 4.150</td><td>3.483 3.833</td><td>0.190 0.059</td><td>0.484 0.820 0.253</td><td>0.370 0.763 0.226</td><td>0.349 0.701 0.225</td><td>0.314 0.647 0.233</td><td>0.277 0.520 0.203</td></tr><tr><td></td><td>Off</td><td>Zero-shot Few-shot Ours</td><td></td><td></td><td></td><td>0.342 0.211 0.081</td><td>0.442 0.812</td><td>0.359 0.700</td><td>0.341 0.624 0.327</td><td>0.284 0.579 0.320</td><td>0.287 0.504</td></tr><tr><td>Gemini On</td><td></td><td>Zero-shot Few-shot Ours</td><td>4.750 4.933 4.633</td><td>4.867 4.900 4.550</td><td>4.850 4.950 4.817</td><td>0.222 0.122 0.045</td><td>0.354 0.347 0.833 0.262</td><td>0.375 0.447 0.781 0.276</td><td>0.412 0.699 0.283</td><td>0.383 0.687 0.296</td><td>0.249 0.320 0.539</td></tr><tr><td>Claude</td><td>Off</td><td>Zero-shot Few-shot Ours Zero-shot</td><td>1.783</td><td>2.083</td><td></td><td>0.252 0.149 0.062</td><td>0.266 0.731 0.121</td><td>0.340 0.776 0.085</td><td>0.365 0.675 0.086</td><td>0.353 0.665 0.084</td><td>0.244 0.330 0.553</td></tr><tr><td></td><td>On Off</td><td>Few-shot Ours Zero-shot</td><td>2.567 3.950</td><td>2.583 3.367</td><td>1.933 2.300 3.117</td><td>0.543 0.336 0.076 0.552</td><td>0.294 0.813 0.122</td><td>0.246 0.719 0.106</td><td>0.233 0.635 0.074</td><td>0.206 0.612 0.082</td><td>0.091 0.166 0.451 0.058</td></tr><tr><td></td><td></td><td>Ours</td><td>Few-shot Extremely Low-Resource Languages</td><td></td><td></td><td></td><td>0.318 0.061</td><td>0.234 0.787</td><td>0.238 0.747</td><td>0.188 0.221 0.713 0.693</td><td>0.158 0.540</td></tr><tr><td>Acehnese</td><td>GPT On</td><td></td><td>Zero-shot Few-shot</td><td>3.300 3.817 4.367</td><td>2.767 3.083</td><td>2.517 2.417</td><td>0.366 0.307</td><td>0.253 0.338</td><td>0.220 0.262</td><td>0.197 0.167 0.212 0.183</td><td>0.157 0.158</td></tr><tr><td></td><td>Off</td><td>Ours Zero-shot Few-shot</td><td></td><td>3.750</td><td>3.133</td><td>0.058 0.377</td><td></td><td>0.810 0.242 0.276</td><td>0.736 0.700 0.233 0.194 0.293 0.227</td><td>0.639 0.177 0.216</td><td>0.454 0.176 0.202</td></tr><tr><td></td><td>Gemini On</td><td>Ours Zero-shot Few-shot</td><td>4.600 4.817</td><td>4.650</td><td>4.017</td><td>0.303 0.095 0.264</td><td></td><td>0.760 0.273</td><td>0.638 0.603 0.295 0.265 0.288 0.324</td><td>0.562 0.252 0.269</td><td>0.436 0.210</td></tr><tr><td></td><td>Off</td><td>Ours Zero-shot Few-shot</td><td>4.733</td><td>4.483 4.583</td><td>4.283 4.483</td><td>0.206 0.072 0.365</td><td></td><td>0.256 0.537 0.587 0.162 0.190</td><td>0.576 0.210 0.250 0.239</td><td>0.564 0.177 0.234</td><td>0.263 0.512 0.177 0.237</td></tr><tr><td>Claude</td><td>On</td><td>Ours Zero-shot Few-shot</td><td>1.317 2.283</td><td>1.133 2.150</td><td>1.367 1.517</td><td>0.276 0.072 0.561 0.395</td><td></td><td>0.146 0.570 0.620 0.126 0.100 0.198 0.145</td><td>0.593 0.076 0.136</td><td>0.601 0.084 0.158</td><td>0.531 0.055 0.151</td></tr><tr><td></td><td>Off</td><td>Ours Zero-shot Few-shot Ours</td><td>3.917</td><td>3.333</td><td>2.717</td><td>0.052 0.546 0.403</td><td></td><td>0.765 0.144 0.193</td><td>0.659 0.665 0.098 0.086 0.143 0.142</td><td>0.611 0.077 0.126</td><td>0.513 0.042 0.139</td></tr><tr><td>Hitu</td><td>GPT On Off</td><td>Ours Zero-shot</td><td>Zero-shot Few-shot</td><td>1.783 2.217 3.200 2.917</td><td>1.550 1.517 2.167 1.800 2.750</td><td></td><td>0.036 0.890 0.672</td><td>0.770 0.015 0.054</td><td>0.720 0.011 0.018</td><td>0.724 0.710 0.007 0.009 0.033 0.054</td><td>0.586 0.005 0.013 0.590 0.459</td></tr></table>

Continued on next page

Table 8 – continued from previous page
<table><tr><td></td><td></td><td></td><td></td><td colspan="3">Fluency ↑</td><td>Off-Target Rate ↓</td><td colspan="5">Lexical Diversity ↑</td></tr><tr><td>Language</td><td>Model</td><td>Thinking</td><td>Prompt</td><td>5</td><td>15</td><td>Para.</td><td></td><td>5</td><td>10</td><td>15</td><td>20</td><td>Para.</td></tr><tr><td></td><td>Gemini</td><td>On</td><td>Zero-shot</td><td>2.583</td><td>2.433</td><td>2.317</td><td>0.695</td><td>0.077</td><td>0.049</td><td>0.049</td><td>0.043</td><td>0.041</td></tr><tr><td></td><td></td><td></td><td>Few-shot</td><td>2.567</td><td>2.683</td><td>2.783</td><td>0.433</td><td>0.149</td><td>0.126</td><td>0.120</td><td>0.127</td><td>0.099</td></tr><tr><td></td><td></td><td></td><td>Ours</td><td>3.583</td><td>3.050</td><td>2.950</td><td>0.039</td><td>0.752</td><td>0.679</td><td>0.664</td><td>0.614</td><td>0.556</td></tr><tr><td></td><td></td><td>Off</td><td>Zero-shot</td><td></td><td></td><td></td><td>0.777</td><td>0.043</td><td>0.031</td><td>0.028</td><td>0.031</td><td>0.026</td></tr><tr><td></td><td></td><td></td><td>Few-shot</td><td></td><td></td><td></td><td>0.642</td><td>0.078</td><td>0.061</td><td>0.066</td><td>0.077</td><td>0.074</td></tr><tr><td></td><td></td><td></td><td>Ours</td><td></td><td></td><td></td><td>0.050</td><td>0.739</td><td>0.671</td><td>0.656</td><td>0.629</td><td>0.552</td></tr><tr><td></td><td>Claude</td><td>On</td><td>Zero-shot</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.881</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td></td><td></td><td></td><td>Few-shot</td><td>2.067</td><td>2.133</td><td>1.767</td><td>0.663</td><td>0.044</td><td>0.036</td><td>0.051</td><td>0.079</td><td>0.025</td></tr><tr><td></td><td></td><td></td><td>Ours</td><td>3.400</td><td>2.633</td><td>2.783</td><td>0.036</td><td>0.779</td><td>0.719</td><td>0.733</td><td>0.680</td><td>0.615</td></tr><tr><td></td><td></td><td>Off</td><td>Zero-shot</td><td></td><td></td><td></td><td>0.879</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td></td><td></td><td></td><td>Few-shot</td><td></td><td></td><td></td><td>0.566</td><td>0.069</td><td>0.065</td><td>0.083</td><td>0.095</td><td>0.027</td></tr><tr><td></td><td></td><td></td><td>Ours</td><td></td><td></td><td></td><td>0.041</td><td>0.790</td><td>0.734</td><td>0.688</td><td>0.707</td><td>0.572</td></tr></table>

Table 8: Fluency, off-target rate, and lexical diversity per language, model, thinking mode, and prompt. Fluency was only human-evaluated for models with thinking mode enabled (— = not evaluated). Best per model and thinking mode in bold.