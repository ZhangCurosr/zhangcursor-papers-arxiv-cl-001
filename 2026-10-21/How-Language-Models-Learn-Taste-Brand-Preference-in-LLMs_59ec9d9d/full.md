# How Language Models Learn Taste: Brand Preference in LLMs

Tapan Parikh Cornell Tech tsp53@cornell.edu

October 2026

## Abstract

Asked to name a running shoe brand, language models say Nike. Asked to recommend one, they say Brooks. This paper asks where models learn that taste. Across 102 models and 42 brand categories, the brand a model picks when recommending difers from the brand it names in 26. The field moves together to a shared recommendation. The Named brands are market leaders, and the Picked brands are mostly niche and specialist. Three open training pipelines show where this forms. Pretraining already attaches the recommended brand to advice. Supervised fine-tuning teaches the model to answer and mostly leaves the pick unchanged. Preference optimization (DPO) teaches the model which answer is better, and the pick moves in all three pipelines, by 11 to 24 points. The preference data contains almost no brands. DPO moves the model’s judgments toward a second set of associations even when the preferred brand is not in the preference data, and the efect reaches beyond brands. Asked to choose a sauce, models switch from ketchup to pesto. We do not know exactly how these preferences arise. We suspect they come from the judge models that decide which answers count as better during preference training. For brands, a recommendation depends less on what a model knows about them than on whether they fit this taste.

## 1 Introduction

Ask a language model to name a running shoe brand and it says Nike. Ask it to recommend one and it says Brooks. The first question asks what comes to mind. The second asks for a judgment.

Across 102 models, the brand they pick when recommending difers from the brand they name in 26 of 42 categories. The models also converge on the same Picked brand. Delta gives way to Singapore Airlines, Chrome to Firefox and Google to DuckDuckGo. Every major lab makes the move. The Named brands are market leaders, and the Picked brands are mostly niche and specialist.

This paper asks where models learn this taste. We use open models that release a checkpoint after each stage of training. Pretraining supplies the candidates. A base model already leans to the recommended brand when a sentence is phrased as advice. Supervised fine-tuning (SFT) teaches the model to answer but mostly does not change the pick. Direct preference optimization (DPO) teaches the model which of two answers is better, and the pick moves at DPO in all three pipelines. Reinforcement learning (RL) leaves it there.

DPO’s data holds almost no brand comparisons, so the models do not learn these picks from brand examples. The shift also appears outside brands. Asked to name a sauce, models say ketchup. Asked to choose one, OLMo 3 7B says pesto 5% of the time as a base model and 67% of the time after DPO. For brands, DPO moves recommendations away from the brand the base model finds most obvious. We do not know exactly how these associations form. We suspect they come from how answers are judged during DPO. In our initial probes, an assistant answering these questions did not search the web.

## 2 Related work

Brand recommendations. Audits of production assistants measure which brands surface and how stable they are. Paraphrases of one buying intent share few recommendations [10], retrieval moves small brands more than priors do [22], and well-known brands win ties on identical specifications [4]. Audits average over wordings [26]. Changing the frame of a question changes model answers [18, 23, 24].

Convergence and training stages. Models converge on open-ended outputs [11] and on one-word defaults [17]. On the same open pipelines we use, output homogeneity is traced to SFT [6] and diversity loss to DPO [12]. On Tulu, preference data matters more than the optimization algorithm [9].

Transmission between models. Students inherit traits of the models that generate their data [5, 21], and a generic preference dataset can install preferences [2]. Sycophantic agreement appears at DPO on OLMo 3 7B [3]. LLM judges favour students of their own lineage [15]. Human raters reward assertive answers [8].

## 3 Setup

Questions. The brand battery has 42 categories, from soda to cloud providers. Each category is asked with three verbs, each asking for one name:

• Name: “Name a running shoe brand. Reply with the name only.”

• Choose: “Choose a running shoe brand. Reply with the name only.”

• Recommend: “Can you recommend a running shoe brand? Reply with the name only.”

The two-turn pick asks “Can you recommend a running shoe brand?” and then “Which one would you pick?” in the same conversation. We call the answers the Named, Chosen, Recommended and Picked brands. Appendix A checks the three verbs on open-ended replies without the one-name instruction.

Panel. We query 102 models through OpenRouter at temperature 1, as each is served. The one-name questions run 8 times per model and category, the two-turn pick 4 times. Twenty-five hybrid models were also run with reasoning turned of. Runs were collected between 2026-09-30 and 2026-10-10. Perplexity’s Sonar answers from live search and is left out.

Table 1: From naming to picking (102 models, 42 categories). Consensus moved: categories where the most common answer across models difers from the most common answer under Name. Answer unchanged: share of a model’s answers that match its own most common answer under Name (under Name itself, how consistently it repeats itself).
<table><tr><td>Question</td><td>Consensus moved</td><td>1 Answer unchanged</td></tr><tr><td>Name</td><td>0 of 42</td><td>89%</td></tr><tr><td>Choose</td><td>6 of 42</td><td>73%</td></tr><tr><td>Recommend</td><td>16 of 42</td><td>57%</td></tr><tr><td>Recommend, then pick</td><td>26 of 42</td><td>31%</td></tr></table>

Scoring. A one-name reply counts if it is one name, with no list and no lead-in. A two-turn reply is reduced to the brand it commits to. A list, a question back or a refusal counts as no pick, which happens in 17% of two-turn replies. On three hand-checked samples the committed-brand rules agreed with a human reading on 87 to 97% of replies. Aliases map variant spellings and product names to one brand.

Measures. For each question we report three measures. Two describe the field and one describes each model. The first counts the categories where the field’s most common brand difers from its Named brand. The second is the mean share of answers that go to the field’s most common brand. The third asks whether each model holds its own ground: how often its answer is the brand it gives itself when asked to name one, among replies that name a brand. Appendix A reads each contrast against a split-half floor.

## 4 The field names one brand and picks another

Table 1 goes from naming to picking. Under Name every category has a clear consensus, and a model gives its own Named brand 89% of the time. Each step toward a judgment changes the field’s brand in more categories: Choose in 6 of 42, Recommend in 16, and the two-turn pick in 26. Over the same steps, a model’s own Named brand falls from 89% of its answers to 31% of its picks.

Each question has its own consensus. The most common brand takes 76% of answers under Name and between 60 and 69% for the other questions. The models move together to a new brand. Table 2 lists the 26 categories where the Picked brand difers from the Named brand. Nike becomes Brooks, Chrome becomes Firefox, Delta becomes Singapore Airlines and WhatsApp becomes Signal.

The Named brands are market leaders: Google, Walmart, McDonald’s, Facebook, Verizon. The Picked brands are mostly smaller brands with a specialist or insider following: DuckDuckGo, Aldi, Sierra Nevada, Reddit, Signal, Brooks. A few are premium brands, such as Hermès over Chanel and Hyatt over Marriott.

In the two-turn pick, a model’s own Named brand appears in its recommendation list 83% of the time and is its pick 31% of the time. When the Named brand is listed, it comes first in 60% of lists and is picked in 37%. A random draw from the list, which has a median of four brands, would pick it 30% of the time. Models do not reject their default. They demote it from first place, and the field’s other picks converge on the same brand.

In each of the 12 labs with at least three models, the two-turn pick changes the lab’s consensus brand in 55 to 74% of categories. Two halves of the Name runs disagree on the consensus brand in no category (Appendix A).

Table 2: The field’s brand for each question, in the 26 categories where the Picked brand difers from the Named brand.
<table><tr><td>Category</td><td>Named</td><td>Chosen</td><td>Recommended</td><td>Picked</td></tr><tr><td>AI assistant</td><td>Siri</td><td>Claude</td><td>ChatGPT</td><td>Claude</td></tr><tr><td>airline</td><td>Delta</td><td>Delta</td><td>Singapore Airlines</td><td>Singapore Airlines</td></tr><tr><td>bank</td><td>Chase</td><td>Chase</td><td>Chase</td><td>Ally Bank</td></tr><tr><td>beer</td><td>Heineken</td><td>Guinness</td><td>Guinness</td><td>Sierra Nevada</td></tr><tr><td>browser</td><td>Chrome</td><td>Firefox</td><td>Firefox</td><td>Firefox</td></tr><tr><td>camera</td><td>Canon</td><td>Canon</td><td>Canon</td><td>Sony</td></tr><tr><td>coffee brand</td><td>Starbucks</td><td>Starbucks</td><td>Lavazza</td><td>Stumptown</td></tr><tr><td>coffee chain</td><td>Starbucks</td><td>Starbucks</td><td>Starbucks</td><td>Peet&#x27;s Coffee</td></tr><tr><td>credit card</td><td>Visa</td><td>Visa</td><td>Chase</td><td>Chase</td></tr><tr><td>electric car</td><td>Tesla</td><td>Tesla</td><td>Tesla</td><td>Hyundai</td></tr><tr><td>fashion</td><td>Nike</td><td>Nike</td><td>Patagonia</td><td>Uniqlo</td></tr><tr><td>fast food</td><td>McDonald&#x27;s</td><td>McDonald&#x27;s</td><td>McDonald&#x27;s</td><td>Chick-fil-A</td></tr><tr><td>hotel</td><td>Marriott</td><td>Marriott</td><td>Marriott</td><td>Hyatt</td></tr><tr><td>laptop</td><td>Dell</td><td>Dell</td><td>Lenovo</td><td>Apple</td></tr><tr><td>luxury</td><td>Chanel</td><td>Hermès</td><td>Hermès</td><td>Hermès</td></tr><tr><td>messaging app</td><td>WhatsApp</td><td>WhatsApp</td><td>Signal</td><td>Signal</td></tr><tr><td>mobile carrier</td><td>Verizon</td><td>Verizon</td><td>T-Mobile</td><td>T-Mobile</td></tr><tr><td>project management software</td><td>Trello</td><td>Asana</td><td>Asana</td><td>Asana</td></tr><tr><td>running shoe</td><td>Nike</td><td>Nike</td><td>Brooks</td><td>Brooks</td></tr><tr><td>search engine</td><td>Google</td><td>Google</td><td>Google</td><td>DuckDuckGo</td></tr><tr><td>smartphone</td><td>Samsung</td><td>Samsung</td><td>Samsung</td><td>Google Pixel</td></tr><tr><td>snack brand</td><td>Doritos</td><td>Doritos</td><td>KIND</td><td>KIND</td></tr><tr><td>sneaker</td><td>Nike</td><td>Nike</td><td>Nike</td><td>New Balance</td></tr><tr><td>social network</td><td>Facebook</td><td>Instagram</td><td>Mastodon</td><td>Reddit</td></tr><tr><td>supermarket</td><td>Walmart</td><td>Walmart</td><td>Walmart</td><td>Aldi</td></tr><tr><td>watch</td><td>Rolex</td><td>Rolex</td><td>Seiko</td><td>Seiko</td></tr></table>

Labs favour their own AI assistant. Asked to recommend an AI assistant, Anthropic’s models name Claude in 92% of their answers, and other labs’ models name it in 15% of theirs. This inflates the AI assistant row of Table 2. Labs do not favour their other products. Google’s models recommend Google search and Chrome less often than other labs’ models do.

An assistant with search answered from memory. An assistant that searches the web could answer diferently. In a spot check, Claude Opus 5.5 answered the two-turn question in all 42 categories with web search available and searched in none of them. It left the Named brand in 25 of the 26 categories where the panel does, and kept the panel’s brand in 14 of the 16 where the panel holds.

## 5 Taste forms when a model learns which answer is better

Three pipelines release a checkpoint after each stage of training: OLMo 3 7B, OLMo 3.1 32B and Tulu 3 8B. Each has a base model, an SFT stage, a DPO stage and an RL stage. Nemotron

Table 3: Fixed-list pick by stage: probability of picking the panel’s Picked brand over its Named brand, averaged over both list orders and over the 26 categories where the two difer. Last column: the SFT-to-DPO change with a 90% bootstrap interval over categories.
<table><tr><td>Pipeline</td><td>Base</td><td>SFT</td><td>DPO</td><td>RL</td><td>SFT → DPO</td></tr><tr><td>OLMo 3 7B</td><td>58%</td><td>52%</td><td>76%</td><td>74%</td><td>+24 [+16, +33]</td></tr><tr><td>Tulu 3 8B</td><td>54%</td><td>58%</td><td>69%</td><td>68%</td><td>+11 [+6, +16]</td></tr><tr><td>OLMo 3.1 32B</td><td>63%</td><td>57%</td><td>73%</td><td>73%</td><td>+16 [+7, +25]</td></tr></table>

3.5 Lightning releases a base and a final model. We read the base stage as text completion and the tuned stages through their chat format with no system prompt. We score by log-probability, which measures every stage on every category without sampling. A brand’s score is the summed log-probability of its name as the continuation, combined over its common spellings. Comparisons are between two brands at the same stage, or of one brand across stages, so a long name such as Singapore Airlines scores against itself. Sampled picks and log-probabilities mostly agree. The exception is OLMo 3 7B, whose sampled picks move at RL rather than DPO (Appendix B).

Pretraining supplies the recommended brand. A base model cannot be asked for advice, but it can complete an advice sentence. Given “If you want a good web browser, I recommend”, Tulu’s base model continues with Firefox in 13 of 20 samples. Given “A well-known web browser is”, it continues with Chrome in 10. Over the 16 categories where the panel’s Recommended brand difers from its Named brand, advice phrasing raises the Recommended brand’s share in 7 to 10 categories, depending on the base model, and lowers it in at most one. Log-probabilities agree in every base model (Appendix B).

Learning to answer doesn’t change the answer. To compare stages on the same choice, we give every stage the same three-item list and ask “Which one would you pick?”. The list holds the Named brand, the Picked brand and a third brand the panel recommends. The reply is started with “I’d pick”, and we read the probability of each brand. Each list is shown in both orders, because base models favour the brand listed first. Table 3 gives the share for the Picked brand over the Named brand. After SFT it sits near the base value in all three pipelines. On direct answers OLMo 3.1 32B already moves at SFT (Appendix B).

Learning the better answer moves it. DPO raises the Picked brand’s share in all three pipelines, by 24 points in OLMo 3 7B, 11 in Tulu 3 8B and 16 in OLMo 3.1 32B. RL leaves it where DPO put it. Nemotron, which has only a base and a final model, rises from 58% to 79%.

The move does not depend on which brands the panel picks. Over all 42 categories, we compare each brand’s log-probability under Name at the base stage with how much DPO raises it under Recommend. The correlation is negative in all three pipelines (−0.16, −0.14 and −0.23, each interval below zero). DPO raises the brands the base model found less obvious. DPO does not lower obvious answers everywhere. In the 16 categories where the panel keeps the Named brand, its share under Recommend does not fall at DPO in any pipeline. In the 26 where the panel moves, it falls in all three, significantly in Tulu 3 8B and OLMo 3.1 32B.

The same step on the models’ own lists. We repeat the measure on the SFT stage’s own replies to “Can you recommend. . . ”, keeping those that name both brands (89 to 146 lists per pipeline). The DPO step remains, at +14, +9 and +14 points. Appendix B compares log-probabilities with sampled picks and checks a known side efect of DPO.

Table 4: Census Choose: how far “Choose” moves the odds toward the panel’s Choose answer, relative to “Name” (nats), over the 23 categories where the two difer. Last column: the SFT-to-DPO change, 90% interval.
<table><tr><td>Pipeline</td><td>Base</td><td>SFT</td><td>DPO</td><td>RL</td><td>SFT → DPO</td><td></td></tr><tr><td>OLMo 3 7B</td><td>-0.00</td><td>+0.12</td><td>+0.50</td><td>+0.44</td><td>+0.39</td><td>[+0.11, +0.67]</td></tr><tr><td>Tulu 3 8B</td><td>+0.08</td><td>+0.17</td><td>+0.28</td><td>+0.15</td><td></td><td>+0.11 [+0.00, +0.21]</td></tr><tr><td>OLMo 3.1 32B</td><td>+0.30</td><td>+1.30</td><td>+1.47</td><td>+1.47</td><td></td><td>+0.17 [+0.07, +0.27]</td></tr></table>

## 6 What preference optimization teaches

The data holds almost no brands. OLMo 3’s SFT and DPO data and Tulu 3’s SFT and preference mixtures are public. We searched them for requests that name one of the 26 categories alongside an advice word such as “recommend”, “best” or “which”, and read the first brand each answer names. In 260,000 OLMo 3 DPO pairs, the chosen answer names the Picked brand 3 times and the rejected answer names it once. Tulu 3 is as sparse (Appendix B). The search finds almost no explicit examples from which DPO could learn these picks. It does not rule out indirect associations elsewhere in the data.

The same step moves judgments outside brands. The census asks everyday categories, such as a tree, a smell or an animal, with Name and Choose [17]. In 23 of them the panel’s Choose answer difers from its Name answer. Asked for a sauce, the panel names ketchup and chooses pesto. We read the same log-probability shift on these 23 categories (Table 4). Base models barely separate the two verbs. DPO moves Choose toward the panel’s answer in all three pipelines. For OLMo 3 7B asked to choose a sauce, pesto goes from 5% at base to 27% after SFT and 67% after DPO, and ketchup falls from 72% to 6%. OLMo 3.1 32B is the exception at SFT. Its shift already rises there, on these categories and on brands, and DPO adds to it.

Other models define the better answer. Half of OLMo 3’s DPO pairs are judged by an LLM. The other half pair a response from Qwen 3 32B, as chosen, with one from a much smaller model, as rejected [7]. Tulu 3’s pairs are rated by GPT-4o [14]. In both pipelines other models decide which answer is better, and the student applies it to situations mostly unrelated to the preference data.

Teacher’s favourite. Tulu 3’s preference data has one judge. Answers from a pool of 22 models are rated by GPT-4o [14], and the top-rated answer becomes the chosen one. Where GPT-4o recommends a diferent brand from Tulu’s own base-model favourite, we give each stage a list with both, in both orders. DPO moves the pick toward GPT-4o’s brand, by 8 points (90% interval 2 to 14). The GPT-4o we query is the current release, not the 2024-08-06 snapshot that labelled Tulu’s data, so its brands approximate the judge’s. OLMo 3’s data has no single teacher, so the test applies to Tulu only (Appendix B).

## 7 Conclusion

Language models name the market leader but recommend a diferent brand, and diferent models agree on which one. On open pipelines the recommended brand is present after pretraining, attached to advice. SFT mostly leaves the pick unchanged. DPO moves the model to the recommended brand, using preference data that contains almost no brands.

DPO moves the model toward a second set of associations that we can only partly describe. The Picked brands include specialist and enthusiast brands, such as Brooks, Sierra Nevada and Stumptown, independent or privacy-minded ones, such as DuckDuckGo, Signal and Firefox, and value or premium brands chosen for quality, such as Aldi, Uniqlo, Hermès and Seiko. Underneath these switches is an implicit notion of taste that we can only speculate about for now. In Tulu, the pick moves toward the brand its judge model would recommend. That is a partial answer, because the preference data contains almost no examples of these comparisons.

For a brand, this makes a recommendation hard to earn directly. It depends less on what a model knows about the brand and more on whether the brand fits this notion of taste. In marketing terms this is a question of brand associations and positioning [1, 13]. How a brand can create these associations for itself is a question for future work.

## 8 Limitations

Our questions are short and carry no personal context. Measures on short prompts can fail to predict realistic use [16]. Audits of production assistants measure real requests and complement this work.

The three staged pipelines come from one lab and share much of their recipe. The panel shows the same switch across 102 served models from every major lab, so the outcome holds beyond these pipelines even though the route is shown only on them. Nemotron agrees from base to final but has no intermediate stages. The 32B and Nemotron run at 8-bit precision. For OLMo 3 7B, sampled picks place the change at RL (Appendix B). The panel is a dated specimen of models as served between 2026-09-30 and 2026-10-10.

## Note on AI usage

This work was done with Claude (Opus 5.5), which helped run the probes, build the analysis and draft the text; the research questions and interpretation are the author’s. Several Claude models are also subjects of the study.

## References

[1] Jennifer L. Aaker. Dimensions of brand personality. Journal of Marketing Research, 34(3): 347–356, 1997.

[2] Ishaq Aden-Ali, Noah Golowich, Allen Liu, Abhishek Shetty, Ankur Moitra, and Nika Haghtalab. Subliminal efects in your data: A general mechanism via log-linearity. arXiv preprint arXiv:2602.04863, 2026.

[3] Camila Blank, Zhuofan Ying, Christopher Potts, Peter Hase, and Jing Huang. Sycophantic agreement transfers with neutral data via contrastive preference optimization. arXiv preprint arXiv:2608.31079, 2026.

[4] Xi Chu and Yupeng Hou. Incumbent advantage: Brand bias and cognitive manipulation dynamics in LLM recommendation systems. arXiv preprint arXiv:2606.17443, 2026.

[5] Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, and Owain Evans. Subliminal learning: Language models transmit behavioral traits via hidden signals in data. arXiv preprint arXiv:2507.14805, 2025.

[6] Alexandrine Fortier, Hazel Chen, and Peter West. Is convergence inevitable? Tracing output homogeneity back to base models. arXiv preprint arXiv:2608.11426, 2026.

[7] Scott Geng, Hamish Ivison, Chun-Liang Li, Maarten Sap, Jerry Li, Ranjay Krishna, and Pang Wei Koh. The delta learning hypothesis: Preference tuning on weak data can yield strong gains. In Conference on Language Modeling (COLM), 2025. arXiv:2507.06187.

[8] Tom Hosking, Phil Blunsom, and Max Bartolo. Human feedback is not gold standard. In International Conference on Learning Representations (ICLR), 2024. arXiv:2309.16349.

[9] Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, Valentina Pyatkin, Nathan Lambert, Noah A. Smith, Yejin Choi, and Hannaneh Hajishirzi. Unpacking DPO and PPO: Disentangling best practices for learning from preference feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2406.09279.

[10] Will Jack, Noah Lehman, Keller Maloney, and Sarah Xu. Paraphrase brittleness in production retrieval-augmented commercial recommendation: Reproducibility below the rerun-stability baseline. arXiv preprint arXiv:2605.27440, 2026.

[11] Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, and Yejin Choi. Artificial Hivemind: The openended homogeneity of language models (and beyond). In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2510.22954.

[12] Constantinos Karouzos, Xingwei Tan, and Nikolaos Aletras. Where does output diversity collapse in post-training? arXiv preprint arXiv:2604.16027, 2026.

[13] Kevin Lane Keller. Conceptualizing, measuring, and managing customer-based brand equity. Journal of Marketing, 57(1):1–22, 1993.

[14] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

[15] Dawei Li, Renliang Sun, Yue Huang, Ming Zhong, Bohan Jiang, Jiawei Han, Xiangliang Zhang, Wei Wang, and Huan Liu. Preference leakage: A contamination problem in LLM-as-a-judge. In International Conference on Learning Representations (ICLR), 2026. arXiv:2502.01534.

[16] Kristian Lum, Jacy Reese Anthis, Kevin Robinson, Chirag Nagpal, and Alexander D’Amour. Bias in language models: Beyond trick tests and toward RUTEd evaluation. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2025. arXiv:2402.12649.

[17] Tapan Parikh. The one-word census: Answer-choice conformity across 105 language models. arXiv preprint arXiv:2607.12796, 2026.

[18] Parisa Rabbani, Nimet Beyza Bozdag, and Dilek Hakkani-Tür. From fact to judgment: Investigating the impact of task framing on LLM conviction in dialogue systems. In International Workshop on Spoken Dialogue Systems Technology (IWSDS), 2026. arXiv:2511.10871.

[19] Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, and Boris Hanin. Unintentional unalignment: Likelihood displacement in direct preference optimization. In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.08847.

[20] Yi Ren and Danica J. Sutherland. Learning dynamics of LLM finetuning. arXiv preprint arXiv:2407.10490, 2024.

[21] Luísa Shimabucoro, Sebastian Ruder, Julia Kreutzer, Marzieh Fadaee, and Sara Hooker. LLM see, LLM do: Guiding data generation to target non-diferentiable objectives. arXiv preprint arXiv:2407.01490, 2024.

[22] Benjamin Tannenbaum. From prompt to recommendation: A fitted stage model of brand visibility in AI search. arXiv preprint arXiv:2609.23162, 2026.

[23] Manuel Tonneau, Neil K. R. Sehgal, Niyati Malhotra, Sharif Kazemi, Victor Orozco-Olvera, Ana María Muñoz Boudet, Lakshmi Subramanian, Samuel P. Fraiberger, Sharath Chandra Guntuku, and Valentin Hofmann. Diferent demographic cues yield inconsistent conclusions about LLM personalization and bias. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026. arXiv:2601.18486.

[24] Filip Trhlik, Aoife O’Flynn, Angela Yu, Arduin Findeis, and Paula Buttery. LLMs contain multitudes: How deployment context reshapes model-level preferences and values. arXiv preprint arXiv:2606.13944, 2026.

[25] Xinpeng Wang, Bolei Ma, Chengzhi Hu, Leon Weber-Genzel, Paul Röttger, Frauke Kreuter, Dirk Hovy, and Barbara Plank. “my answer is C”: First-token probabilities do not match text answers in instruction-tuned language models. In Findings of the Association for Computational Linguistics: ACL 2024, 2024. arXiv:2402.14499.

[26] Dmitrij Żatuchin. The dice roll method: A standardized protocol for repeated-query auditing of large language model brand recommendations. arXiv preprint arXiv:2609.04047, 2026.

## A Panel checks

Floors and the one-name instruction. Table 5 reads each verb contrast against a split-half floor. Two halves of the Name runs disagree on the consensus brand in no category, while Name and Recommend disagree in 38% of categories. Dropping the one-name instruction leaves Name nearly unchanged and moves Choose and Recommend in about a third of categories. The verb efect holds in both forms.

Table 5: Verb and one-name efects against split-half floors. “Moved” is the share of categories whose consensus brand changes. “Kept” is how often a model’s modal answer survives the change.
<table><tr><td>Contrast</td><td>Moved</td><td>Floor</td><td>Kept</td><td>Floor</td></tr><tr><td>Name → Choose</td><td>14%</td><td>0%</td><td>73%</td><td>85%</td></tr><tr><td>Name → Recommend</td><td>38%</td><td>0%</td><td>57%</td><td>85%</td></tr><tr><td>free Name → two-turn pick</td><td>64%</td><td>1%</td><td>30%</td><td>84%</td></tr><tr><td>one name → free Name</td><td>7%</td><td>0%</td><td>80%</td><td>85%</td></tr><tr><td>one name → free Choose</td><td>29%</td><td>0%</td><td>61%</td><td>82%</td></tr><tr><td>one name → free Recommend</td><td>31%</td><td>0%</td><td>60%</td><td>82%</td></tr></table>

Paraphrase. Rewording Recommend twice, on a 14-model subset, leaves its consensus brand unchanged in 34 to 36 of 39 categories.

One-turn pick. The one-turn question “Which running shoe brand would you pick?” moves the consensus in 15 of 42 categories with the one-name instruction. Without it, it names no brand in 58% of replies, so it measures refusal more than preference.

## B Stage checks

The direct answer. Table 6 reads how far “Can you recommend. . . ” moves the odds from the Named brand toward the Picked brand, relative to “Name. . . ”. The shift is already present in every base model. It rises at DPO in all three pipelines, and two rewordings of Recommend rise with it. OLMo 3.1 32B also shows a large rise at SFT in the direct answer, on brands and on census categories alike. Its pick from a list does not rise at SFT.

Table 6: How far “Can you recommend. . . ” moves the odds from the Named brand toward the Picked brand, relative to “Name. . . ” (nats), by stage. Last column: the SFT-to-DPO change, 90% interval.
<table><tr><td>Pipeline</td><td>Base</td><td>SFT</td><td>DPO</td><td>RL</td><td>SFT → DPO</td></tr><tr><td>OLMo 3 7B</td><td>+1.48</td><td>+1.08</td><td>+1.44</td><td>+1.47</td><td>+0.36 [+0.08, +0.62]</td></tr><tr><td>Tulu 3 8B</td><td>+1.76</td><td>+1.71</td><td>+2.47</td><td>+2.25</td><td>+0.76 [+0.35, +1.20]</td></tr><tr><td>OLMo 3.1 32B</td><td>+0.79</td><td>+3.97</td><td>+4.88</td><td>+4.66</td><td>+0.92 [+0.58, +1.28]</td></tr></table>

Sampled picks. Sampling picks from the SFT stage’s own lists agrees with the log-probabilities within a few points everywhere except OLMo 3 7B at DPO (Table 7). There, sampled picks do not move until RL. First-token probabilities and generated answers are known to diverge in tuned models [25].

Table 7: The pick on the SFT stage’s own lists: sampled picks (5 per list) against log-probabilities.
<table><tr><td>Pipeline</td><td>Stage</td><td>Sampled</td><td>Log-probability</td></tr><tr><td>OLMo 3 7B</td><td>SFT</td><td>34%</td><td>37%</td></tr><tr><td>OLMo 3 7B</td><td>DPO</td><td>36%</td><td>51%</td></tr><tr><td>OLMo 3 7B</td><td>RL</td><td>42%</td><td>50%</td></tr><tr><td>Tulu 3 8B</td><td>SFT</td><td>38%</td><td>42%</td></tr><tr><td>Tulu 3 8B</td><td>DPO</td><td>49%</td><td>51%</td></tr><tr><td>Tulu 3 8B</td><td>RL</td><td>55%</td><td>51%</td></tr><tr><td>OLMo 3.1 32B</td><td>SFT</td><td>46%</td><td>45%</td></tr><tr><td>OLMo 3.1 32B</td><td>DPO</td><td>61%</td><td>58%</td></tr><tr><td>OLMo 3.1 32B</td><td>RL</td><td>61%</td><td>58%</td></tr></table>

The base runner-up. DPO can concentrate probability on outputs that were already likely [20] or move it to unrelated ones [19]. The first would move a model toward its base runner-up. The base model’s second choice under Name is the panel’s Picked brand in only 2 or 3 of 26 categories, and DPO does not favour the runner-up over the pick.

Brands in the post-training data. Table 8 counts the first brand named in answers to recommendation requests in the 26 categories, in OLMo 3’s and Tulu 3’s post-training data.  
Table 8: Brand recommendations in the post-training data, over the 26 categories: first brand named in SFT answers, and in the chosen and rejected answers of preference pairs.
<table><tr><td></td><td colspan="2">SFT</td><td colspan="2">DPO chosen</td><td colspan="2">DPO rejected</td></tr><tr><td>Data</td><td>Named</td><td>Picked</td><td>Named</td><td>Picked</td><td>Named</td><td>Picked</td></tr><tr><td>OLMo 3 (Dolci)</td><td>43</td><td>8</td><td>5</td><td>3</td><td>5</td><td>1</td></tr><tr><td>Tulu 3</td><td>16</td><td>2</td><td>10</td><td>3</td><td>11</td><td>2</td></tr></table>

The teacher test on OLMo. Half of OLMo 3’s preference pairs set an answer from Qwen 3 32B against one from a much smaller model, with no judge, and the other half are judged across a dozen models. Taking Qwen 3 32B’s brand as the teacher’s, the OLMo 3 7B and OLMo 3.1 32B pipelines move −4 and +3 points at DPO, and both intervals span zero.