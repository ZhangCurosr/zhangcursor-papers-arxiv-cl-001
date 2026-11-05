# Adapting Qwen3 to Tajik from 0.6B to 8B: Lower Text Loss Did Not Mean Better Understanding

Tohir Saidzoda and Madina Abdusalomzoda and Nasimjon Rajabov Akhrorkhon Ikromov and Komron Giyosiev and Abubakr Abdullozoda Amin Jalilov and Jonibek Kishvarov and Rahmonali Sharipov and Mariyamkhon Shodyeva Saidzoda AI Lab, Saidzoda Engineering Company, Dushanbe, Tajikistan tohir@saidzoda.com

## Abstract

Small LoRA adapters trained on Tajik news and books lowered Qwen3’s Tajik text loss at every size we tried, from 0.6B to 8B, by 0.60 to 1.45 nats per token, but none of them improved Tajik reading comprehension where it can be measured. On Belebele the 1.7B and 8B adapters left accuracy unchanged and the 4B adapter lowered it by 3.7 points; at 0.6B the score rose by 4.2 points but sits near chance and coincides with a change in which answer letter the model prefers, so we do not read it as understanding. At 8B, continued pretraining on about 375 million tokens of raw Tajik text was followed by a Belebele gain of 9.9 points. After a further supervised fine-tuning stage, our main model, e3, is 11.4 points above Qwen3- 8B on Belebele; its English MMLU changed by −1.2 points, and a loss larger than 3.3 points is excluded. The same heavy recipe applied to Qwen3-0.6B-Base brought its text loss close to that of the 8B model after books-only pretraining, while its multiple-choice answers stayed near chance and its English MMLU fell by 7.6 points. We report 15 trained models and their 5 starting models from June to October 2025, scored in 2026 on one harness with paired intervals, and we report what went wrong: lost training records, a thinking-mode dataset with almost no reasoning in it, later branches whose training data was never documented, and a 4-bit release that contains no transformer layers.

## 1 Introduction

Tajik, the state language of Tajikistan, is written in a Cyrillic alphabet with six letters that Russian does not have. Open language models rarely list it among their supported languages. In June 2025 we set out to make Qwen3 (Yang et al., 2025) usable in Tajik with the resources a small lab has: single GPUs, a few days per run, and Tajik text collected from news sites, books and Wikipedia.

We tried two recipes. The light recipe, used first, trains a small attention-only LoRA adapter (Hu et al., 2021) on a 4-bit model (Dettmers et al., 2023) with at most about 21.4 million tokens of news and books per epoch. The heavy recipe adds continued pretraining (CPT) on about 375 million tokens of raw Tajik text, followed by supervised fine-tuning (SFT) on more than a million Tajik chat examples. We applied the light recipe at 0.6B, 1.7B, 4B and 8B, and the heavy recipe at 8B and later again at 0.6B.

In 2025 we judged these models by reading their answers. They all wrote Tajik, so all of them looked like progress. None was scored on a held-out benchmark at the time. For this report we scored the 15 trained models and their 5 starting models in 2026 on one harness, each trained model against the model it started from, and the numbers tell a different story from the one we believed.

Our contributions are:

1. A measured record of five months of Tajik adaptation. Models from 0.6B to 8B, scored on Tajik reading comprehension, English knowledge, and Tajik text loss, with paired intervals against each model’s own starting point (Section 6).

2. Lower text loss is not understanding. Light adapters lower Tajik text loss at every size without improving reading comprehension at 1.7B, 4B or 8B; the 0.6B scores sit near chance. Only the 8B models with continued pretraining improved on it (Figure 1). At 0.6B even the heavy recipe lowered text loss without moving the multiple-choice scores away from chance, and we show that the scores of the smallest models are dominated by a preference for one answer letter (Table 3).

3. A record of what failed. Lost epochs, a thinking-mode dataset that was 99.5% ordinary chat, a tax test set deleted before it was used, later branches whose training data was never documented, and a quantised release missing its transformer layers (Section 7).

## 2 Related Work

Adapting models to a new language. The usual route is continued pretraining on raw text in the new language, often followed by instruction tuning (Cui et al., 2023). Continued pretraining shifts the model’s distribution and can make it forget what it knew (Luo et al., 2023); learning-rate rewarming and replay reduce the loss (Gupta et al., 2023; Ibrahim et al., 2024). LoRA (Hu et al., 2021) and its 4-bit form QLoRA (Dettmers et al., 2023) make this affordable on one GPU. LoRA learns less from new data than full fine-tuning but also forgets less (Biderman et al., 2024), which matters when the goal is a new language rather than a new style. Multilingual pretraining helps low-resource languages up to a point and then competes with them (Chang et al., 2023).

Tajik. Large web corpora now include Tajik (Kudugunta et al., 2023; Penedo et al., 2025), and recent work has released Tajik text-processing tools and compared parameter-efficient fine-tuning methods on Tajik (Arabov et al., 2026; Arabov, 2026). Belebele (Bandarkar et al., 2024), built on FLO-RES passages (Team et al., 2022), covers Tajik reading comprehension.

Evaluating adapted models. Studies of language adaptation usually report held-out loss and downstream accuracy together. Our results add a case where the two disagree strongly: text loss improves in every run, while accuracy, measured by Belebele, improves only when the model is large enough and the training text is large enough.

Quantisation. SINQ quantises weights to 4 bits without calibration data by balancing row and column scales (Müller et al., 2025). We used it in 2025 and report in Section 7 why our quantised uploads cannot be evaluated.

## 3 Training Data

News and books (light recipe). 26,601 news articles and 15,276 book chunks, 41,877 documents in total. Each document was wrapped as one chat turn, with the title as the user message and the text as the answer, and the loss covered the whole text. This is continued pretraining inside a chat template, not instruction tuning. Texts were cut to 512 tokens for the 1.7B and 4B runs, which gives at most about 21.4 million tokens per epoch. The data of the 0.6B adapter we evaluate was not recorded; an earlier 0.6B run used news only, cut to 1,024 tokens.

CPT corpus (heavy recipe). 183,050 chunks of 2,048 tokens each, about 375 million tokens: 39,100 from books, 128,463 from news and 15,487 from Tajik Wikipedia. The text is raw, cut into fixed windows without document boundaries. The Wikipedia chunks still contain page navigation text. One epoch of this corpus is about 18 times one epoch of the light-recipe data.

SFT data. For 8B: 1,227,427 chat examples from 11 files, including question answering, dictionary entries, Wikipedia rewritten as dialogue, sentence tasks, instructions and official documents. It was called a “thinking mode” dataset, but only about 6,600 examples (0.5%) contain reasoning. For 0.6B: 1,122,175 examples from 12 sources, which overlap with the 8B set but drop the Wikipedia dialogues and add university entrance-exam questions and questions about Tajikistan.

Branch data. Three later 8B branches added: a maths set of 17,446 examples whose contents were never documented; 10,982 university entranceexam questions with model-written explanations and 8,748 questions about Tajikistan; and 70,186 examples on the Tajik tax code (45.8 million tokens).

## 4 Models and Recipes

All training used Unsloth with LoRA adapters. Table 1 summarises the four groups.

Light recipe, June 2025. Qwen3-0.6B, Qwen3- 1.7B and Qwen3-4B (the released chat models) were loaded in 4 bits and given attention-only adapters of rank 8 to 16 on the news and books data. The 1.7B model has two versions: v1 after 41,877 documents, and v2 after a further pass over a subset. An adapter of rank 8 with the same settings as the 4B run was trained on Qwen3-8B; its training data was not recorded.

Heavy recipe at 8B, July to August 2025. We first ran CPT on books only, with a rank-32 adapter on the 16-bit model for 3 epochs (logged training loss 2.73 at the start and 1.73 at step 3,650). Separately, a rank-16 adapter on all projections of the 4- bit model was trained for 4 epochs on the full CPT corpus; only the last epoch has a training record, and the middle epochs are known only from saved checkpoints. 3 attempts at full fine-tuning on the books data were stopped within the first steps and nothing was saved. SFT then continued the same adapter for 3 epochs on the 1,227,427 chat examples, and the result was merged into Qwen3-8B as e3 (SaidzodaEng/Ameena\_Qwen3-8B\_e3). Three branches continue from there: maths (e3.1), a further adapter on the merged e3.1 model with the entrance-exam and Tajikistan data (MultiStage), and the tax code. Because all 8B stages continue one adapter (the maths notebook loads the SFT adapter as its starting point), the e3.1 adapter applied to Qwen3-8B and the merged e3.1 file should be the same model; their answers are identical on every test.

Heavy recipe at 0.6B, September to October 2025. We returned to the smallest size with Qwen3-0.6B-Base and a rank-64 adapter on all projections: SFT for 3 epochs, SFT for about 5 more (stopped just before the end), CPT on the same corpus as the 8B model for 2 epochs, and a final SFT of 2 epochs that started from a checkpoint of an earlier, stopped attempt (SaidzodaEng/ Ameena\_Qwen3-0.6B\_LoRA\_e8\_p2\_e3). In the SFT stages the loss covered only the answers. The mask also excluded the end-of-turn token, so the model never learned to stop, which may explain the endless replies we saw in its samples.

## 5 Evaluation

No 2025 model had a held-out score, so we scored all of them in October 2026 with one script on one NVIDIA A10.

Tests.

• Belebele Tajik (Bandarkar et al., 2024): 900 reading-comprehension questions about short passages. This is our measure of understanding.

• English MMLU (Hendrycks et al., 2021): 1,003 of the 14,042 test questions (every 14th, which keeps all subjects in proportion), to see what the models forgot.

• Tajik text loss: the mean negative log-likelihood per token over the 488 distinct Tajik Belebele passages (139,477 tokens). All models share the Qwen3 tokenizer, so the values are comparable. This is our measure of fluency.

Qwen3-8B was scored on the full MMLU set; because each question is scored on its own, taking the same subset from its saved per-question answers gives exactly the result of scoring only those questions.

Protocol. Every multiple-choice question is shown as plain text with the letters A to D, and the answer is the letter with the highest next-token probability. No chat template is used for any model, so chat models and base models are scored the same way. Most adapters were trained on 4-bit models (the books-only CPT adapter and the first 0.6B SFT stage used 16-bit models); all are applied to the 16- bit model they were trained from. For every trained model we compare its answers with those of its starting model on the same questions, and report a paired 95% bootstrap interval over questions. Each model was trained once, so these intervals reflect test sampling only, not training variation.

Letter preference. Scoring by the most likely letter is known to be sensitive to a model’s preference for particular letters (Zheng et al., 2023). The correct answers in Belebele are spread close to evenly over the four letters (at most 28% on one letter), so for each model we also report how often it chose its most frequent letter (Table 3). The 8B models choose their most frequent letter on at most 40% of the Belebele questions, but stock Qwen3-0.6B chooses “A” on 95% of the Belebele questions, and the June 0.6B adapter chooses “C” on 45% of the English MMLU questions. For such models a change in accuracy mostly reflects a change in letter preference, and we do not read it as a change in understanding. We stored only the chosen letter, not the four probabilities, so we cannot correct for this preference after the fact.

## 6 Results

Table 2 gives all scores, Figure 1 plots text loss against Belebele accuracy, and Table 3 shows how strongly each model prefers one answer letter.

## 6.1 Light adapters: lower text loss, no better comprehension

Every light adapter lowered text loss: by 1.45 at 0.6B, 1.29 at 1.7B, 0.86 at 4B and 0.60 at 8B. None improved Belebele at 1.7B, 4B or 8B. At 1.7B the two versions changed it by −0.8 and +0.0 points, at 8B by +0.1 (interval −3.3 to +3.2), and at 4B it fell by 3.7 points (interval −6.9 to −0.2). At 0.6B the adapter raised Belebele by +4.2 points (interval +0.9 to +7.8), but both models are near chance and the stock model chose one letter on 95% of the questions against 48% after training (Table 3), so the change is a change in letter preference, not evidence of understanding. English MMLU fell by 3.3 and 3.7 points for the two 1.7B versions and by 9.4 at 0.6B, where the letter preference also changed.

<table><tr><td></td><td>Light, small (June)</td><td>Light, 8B (June)</td><td>Heavy, 0.6B (Sep to Oct)</td><td>Heavy, 8B (Jul to Aug)</td></tr><tr><td>Start model Text</td><td>Qwen3 0.6B, 1.7B, 4B news 41,877 documents</td><td>Qwen3-8B and books, not recorded</td><td>Qwen3-0.6B-Base raw text</td><td>Qwen3-8B 1,122,175 chat examples + raw text, then 1,227,427 chat examples</td></tr><tr><td>Tokens</td><td>(0.6B: not recorded) per about 21.4M</td><td>not recorded</td><td>about 375M (CPT stage)</td><td>about 375M (CPT stage)</td></tr><tr><td>epoch Order</td><td>one adapter run per not recorded</td><td></td><td>SFT, SFT, CPT, SFT</td><td>CPT (4 epochs), then SFT</td></tr><tr><td>Loss on</td><td>model whole text</td><td>not recorded</td><td>answer only</td><td>(3) whole text</td></tr><tr><td>Adapter</td><td>rank 8 to 16, attention rank 8, attention</td><td></td><td>rank 64, all projections</td><td>rank 16, all projections</td></tr><tr><td>Precision</td><td>only 4-bit</td><td>only 4-bit</td><td>16-bit, later 4-bit</td><td>4-bit</td></tr></table>

Table 1: The four groups of models compared in this report. The groups differ in more than size and recipe, so the comparison is not controlled. Values come from the training notebooks and adapter files (Section 4).

![](images/b136031272b9c604727de497765f8ec49a61f0dbd59139acf167470d839ceb80.jpg)  
Figure 1: Tajik text loss (lower is better, axis reversed) against Belebele Tajik accuracy. The dotted line is chance. Every trained model has a lower text loss than its starting model; Belebele rose clearly only for the 8B models with continued pretraining. The 0.6B scores near chance mostly reflect a preference for one answer letter (Table 3).

In 2025 we judged these models by whether they wrote Tajik, and they did. The scores show that writing Tajik and understanding it are different things, and the light recipe changed only the first.

## 6.2 8B: comprehension rose with continued pretraining

After CPT on books alone, Belebele was 7.4 points higher than for Qwen3-8B (interval +4.2 to +10.7), and after CPT on the full corpus 9.9 points higher (interval +6.3 to +13.4). With SFT on top, e3 scores 69.3 on Belebele against 57.9 for Qwen3-8B, with an interval above zero. Its English MMLU changed by −1.2 points (interval −3.3 to +0.9): a loss of more than 3.3 points is excluded, a smaller one is not.

The June light adapter and the CPT adapters start from the same model, but they differ in training text, adapter size and precision, and the June adapter’s data was not recorded. We therefore cannot say which of these differences made the CPT models better; we can only say that comprehension rose in the runs with CPT and not in the run without it.

## 6.3 0.6B: lower text loss, answers still near chance

Qwen3-0.6B-Base with the full heavy recipe reached a text loss of 1.95, close to the 1.83 of the 8B model after books-only CPT and above the 1.58 after CPT on the full corpus. Its Belebele accuracy stayed near chance (30.9), and it chose its most frequent letter on 55% of the Belebele questions. English MMLU fell by 6.2 to 8.5 points at every stage, and the letter preference on Belebele grew at the CPT and final stages. We see no evidence that the 0.6B model understands Tajik questions. Its answers in 2025, which we did not score, pointed the same way: fluent Tajik with wrong facts, such as naming Dushanbe the capital of Japan.

## 6.4 Later 8B branches

The maths model (e3.1), trained on top of e3 with a file named as exam test data whose contents were never documented, changes Belebele by +0.2 points against e3 and lowers English MMLU by 2.7 points. The MultiStage model, trained on top of e3.1 with entrance-exam and Tajikistan questions, changes Belebele by +0.8 points against e3 and English MMLU by −1.5. Neither branch improves reading comprehension beyond e3 by more than the noise, and both have a lower English MMLU point estimate than e3 (we did not compute paired intervals against e3), so we do not count them as progress. The tax-code branch, which saw neither the maths nor the exam data, is 3.0 points below e3 on Belebele and 0.9 points below it on English MMLU.

<table><tr><td>Model</td><td>Text loss ↓ (488 passages)</td><td>Belebele Tajik (900)</td><td>English MMLU (1,003)</td></tr><tr><td colspan="4">Light recipe, June 2025 (attention-only LoRA, at most about 21.4M tokens per epoch)</td></tr><tr><td>Qwen3-0.6B</td><td>4.25</td><td>23.1</td><td>41.5</td></tr><tr><td>+ light adapter (QwenTg-v5)</td><td>2.80 (−1.45)</td><td>27.3 (+4.2*)</td><td>32.1 (−9.4*)</td></tr><tr><td> $\mathbf { Q } \mathrm { w e n } \bar { 3 } \ – 1 . 7 \mathbf { B }$ </td><td>3.69</td><td>34.1</td><td>58.0</td></tr><tr><td>+ light adapter v1</td><td>2.40 (−1.29)</td><td>33.3 (−0.8)</td><td>54.7 (−3.3*)</td></tr><tr><td>+ light adapter v2</td><td>2.46 (−1.23)</td><td>34.1 (+0.0)</td><td>54.3 (−3.7*)</td></tr><tr><td>Qwen3-4B</td><td>3.14</td><td>48.6</td><td>68.3</td></tr><tr><td>+ light adapter</td><td>2.27 (−0.86)</td><td>44.9 (−3.7*)</td><td>67.9 (−0.4)</td></tr><tr><td colspan="4">Heavy recipe at 0.6B, September to October 2025</td></tr><tr><td>Qwen3-0.6B-Base</td><td>3.83</td><td>30.0</td><td>48.8</td></tr><tr><td>+ SFT, 3 epochs</td><td>2.13 (−1.70)</td><td>31.4 (+1.4)</td><td>42.6 (−6.2*)</td></tr><tr><td>+ SFT, 8 epochs</td><td>2.20 (−1.63)</td><td>28.9 (−1.1)</td><td>40.6 (−8.2*)</td></tr><tr><td>+ CPT, 2 epochs</td><td>1.90 (−1.94)</td><td>32.1 (+2.1)</td><td>40.3 (−8.5*)</td></tr><tr><td>+ SFT, 2 epochs (final)</td><td>1.95 (−1.88)</td><td>30.9 (+0.9)</td><td>41.2 (−7.6*)</td></tr><tr><td colspan="4">8B, June to October 2025</td></tr><tr><td>Qwen3-8B</td><td>2.74</td><td>57.9</td><td>73.5</td></tr><tr><td>+ light adapter (June)</td><td>2.13 (−0.60)</td><td>58.0 (+0.1)</td><td>74.0 (+0.5)</td></tr><tr><td>+ CPT, books only</td><td>1.83 (−0.91)</td><td>65.3 (+7.4*)</td><td>75.4 (+1.9*)</td></tr><tr><td> $+ \mathrm { C P T , b o o k s , w i k i , n e w s }$ </td><td>1.58 (−1.16)</td><td>67.8 (+9.9*)</td><td>73.2 (−0.3)</td></tr><tr><td> $+ \mathrm { C P T } + \mathrm { S F T } \left( \mathrm { e } 3 \right)$ </td><td>1.63 (−1.11)</td><td>69.3 (+11.4*)</td><td>72.3 (−1.2)</td></tr><tr><td> $+ \mathrm { C P T } + \mathrm { S F T } + \mathrm { m a t h s } \ ( \mathrm { e } 3 . 1 )$ </td><td>1.82 (−0.91)</td><td> $6 9 . 6 \left( + 1 1 . 7 ^ { * } \right)$ </td><td>69.6 (−3.9*)</td></tr><tr><td> $+ \mathrm { C P T } + \mathrm { S F T } + \mathrm { m a t h s } , \mathrm { m e r g e d f i l e }$ </td><td>1.82 (−0.91)</td><td> $6 9 . 6 \left( + 1 1 . 7 ^ { \ast } \right)$ </td><td>69.6 (−3.9*)</td></tr><tr><td> $+ \mathrm { C P T } + \mathrm { S F T } + \mathrm { m a t h s } + \mathrm { e x a m ~ a n d ~ T a j i k i s t a n }$ </td><td>1.74 (−1.00)</td><td> $7 0 . 1 \ : ( + 1 2 . 2 ^ { * } )$ </td><td>70.8 (−2.7*)</td></tr><tr><td> $+ \mathrm { C P T } + \mathrm { S F T } + \mathrm { t a x } \mathrm { c o d e }$ </td><td>1.68 (−1.05)</td><td>66.3 (+8.4*)</td><td>71.4 (−2.1)</td></tr></table>

Table 2: Results for all 21 rows (the two e3.1 rows are the same model, as an adapter and as a merged file). Text loss is the mean negative log-likelihood per token on the Tajik Belebele passages (lower means the model finds real Tajik text more predictable). The other two columns are accuracy (%) on four-option questions, where guessing scores 25. English MMLU uses 1,003 questions (every 14th item). In brackets: change against the stock model at the top of each block (the unindented row), on the same questions; <sup>∗</sup> marks changes whose paired 95% bootstrap interval excludes zero.

## 7 What Went Wrong

We report these because a small lab is likely to repeat them.

No held-out test in 2025. Every 2025 model was judged by reading its generated answers. Those answers showed Tajik letters and Tajik words, which every model produced, so the light adapters looked as successful as the 8B model. The tax-code data had 3,510 questions set aside for testing, but they were deleted before training and never used.

Lost training records. CPT epochs one to three of the 8B model have no loss record; only the fourth epoch survives, and its loss is flat (0.96 to 0.95) because the adapter was already trained. The training runs behind the June 8B adapter, the books-only CPT adapter and the merged e3.1 model left no notebook, and the June 0.6B adapter left no record of its data. We reconstructed the history in this report from notebooks, Hugging Face upload times and adapter files.

A thinking-mode dataset without thinking. The 8B SFT data was built to teach step-by-step reasoning, but only 0.5% of it contains reasoning. All 698 answers we generated from the trained models start with an empty reasoning block. This is weak evidence on its own, since the chat template can insert such a block, but it agrees with the data.

A 4-bit release without layers. In October 2025 we quantised e3 with SINQ and A-SINQ (Müller et al., 2025) and uploaded the results. Each upload holds 147 tensors (2.49 GB): the input and output embeddings and the normalisation weights. The transformer layers are missing; the most likely cause is that the quantised model was saved with the standard save call of the training library, which does not store the quantised layer type. The size reduction noted at the time measured the embeddings alone, and no quality test was run. These files cannot be evaluated, and we do not report a 4-bit result.

<table><tr><td>Model</td><td>Belebele Tajik English MMLU</td></tr><tr><td>Qwen3-0.6B</td><td>95</td></tr><tr><td>+ light adapter (QwenTg-v5)</td><td>48</td></tr><tr><td>Qwen3-1.7B</td><td>37</td></tr><tr><td>+ light adapter v1</td><td>35 46 32 32</td></tr><tr><td>+ light adapter v2</td><td>49</td></tr><tr><td>Qwen3-4B</td><td>36</td></tr><tr><td>+ light adapter</td><td>37</td></tr><tr><td>Qwen3-0.6B-Base</td><td>48</td></tr><tr><td>+ SFT, 3 epochs</td><td>43</td></tr><tr><td>+ SFT, 8 epochs</td><td>53</td></tr><tr><td>+ CPT, 2 epochs</td><td>56</td></tr><tr><td>+ SFT, 2 epochs (final)</td><td>55</td></tr><tr><td>Qwen3-8B</td><td>33</td></tr><tr><td>+ light adapter (June)</td><td>29</td></tr><tr><td>+ CPT, books only</td><td>31</td></tr><tr><td>+ CPT, books, wiki, news</td><td>36</td></tr><tr><td>+ CPT + SFT (e3)</td><td>31</td></tr><tr><td>+ CPT + SFT + maths (e3.1)</td><td>32</td></tr><tr><td>+ CPT + SFT + maths + exam and Tajikistan</td><td>31</td></tr><tr><td>+ CPT + SFT + tax code</td><td>40</td></tr></table>

Table 3: Share (%) of questions on which each model chose its most frequent letter. The correct answers are close to evenly spread (at most 28% on one letter in Belebele), so a share far above a quarter means the model is mostly choosing by position.

## 8 Discussion

Why text loss fell everywhere. Lowering text loss on Tajik requires the model to predict Tajik word forms and common phrases, which a small adapter can learn from tens of millions of tokens. Answering a question about a Tajik passage also requires the model to connect the Tajik words to what it already knows. Our results suggest that the light adapters taught the first and not the second, and at the smaller sizes they cost English knowledge as well.

Size and recipe. The same CPT corpus and similar SFT data were followed by better comprehension at 8B and not at 0.6B. Qwen3-0.6B-Base scores 48.8 on English MMLU against 73.5 for Qwen3-8B, so it has far less knowledge to carry over into Tajik, and our training reduced its English score further. We read this as a hypothesis, not a finding: the comparison has only two sizes for the heavy recipe and is not controlled (Table 1). The 0.6B run started from the base model rather than the chat model, trained on answers only, and used a larger adapter, and the 1.7B and 4B models never received the heavy recipe.

What we would do differently.

• Hold out a test set before training, and score every checkpoint on it, not only on generated samples.

• Measure comprehension and text loss separately; a falling text loss says nothing about reading comprehension.

• Store the probability of every answer option, so that a preference for one letter can be measured and corrected.

• Check that a quantised file loads and scores before releasing it.

## 9 Release

The starting models are the Qwen3 releases. All trained models in Table 2 are on Hugging Face under SaidzodaEng, except the June 8B adapter, which is private. The main 8B model is SaidzodaEng/Ameena\_Qwen3-8B\_e3 and the final 0.6B adapter is SaidzodaEng/ Ameena\_Qwen3-0.6B\_LoRA\_e8\_p2\_e3. Some repository names do not match their contents; for example, the repository named as the merged MultiStage model holds the same weights as e3, and the quantised repositories hold no layers (Section 7). The evaluation script, the perquestion answers of every model on the three tests of this report, and the scripts that build every number, table and figure in this report are available at https://huggingface.co/datasets/

Tohirju/Saidzoda-Qwen3-Tajik-Evals.

## 10 Conclusion

We scored the Qwen3 models we adapted to Tajik between June and October 2025. Small adapters lowered Tajik text loss at every size and improved reading comprehension at no size where the score is above chance. At 8B, Belebele rose by 9.9 points in the run with continued pretraining on about 375 million tokens, and our main model, e3, is 11.4 points above Qwen3-8B on Belebele with no large loss in English. At 0.6B the heavy recipe lowered text loss while the answers stayed near chance. The lesson we take is to measure comprehension, not only fluency, from the first run.

## Limitations

Not a controlled comparison. The four groups differ in start model (chat model or base model), data, adapter size, precision and loss masking (Table 1). The training data of the June 8B and 0.6B adapters is unknown. Our conclusions about size and recipe rest on consistent patterns, not on runs that change one factor at a time.

Single runs. Each model was trained once, and the intervals cover only the sampling of test questions. Most adapters were trained on 4-bit models and all were evaluated on 16-bit ones.

Scoring. All models were scored by the most likely answer letter, without a chat template. Understanding is measured by one test, Belebele Tajik, a translated reading task; we have no Tajik knowledge or generation test in this report. Chat models may score differently in their own format, and the light adapters were trained inside a chat template, so part of any change may be a change in format. The smallest models prefer one letter strongly (Table 3); we kept only the chosen letter, so we cannot correct for this, and their accuracies near chance carry little information. English MMLU uses 1,003 of the 14,042 questions, and generation quality was not measured.

Possible overlap with tests. We did not check any training data against the test sets. The CPT corpus includes Tajik Wikipedia and the SFT data includes question answering and Wikipedia rewritten as dialogue; these could overlap with Belebele passages, which would make the gains of e3 and the text loss of the CPT models optimistic. The Belebele passages come from FLORES news and travel text, which is not part of our corpora by design, but we did not run a check.

Reconstructed history. Several training records are lost (Section 7). Where a setting is not recorded, we say so rather than infer it. Some repository names on Hugging Face do not match their contents (Section 9).

## Use of AI Assistants

We used AI assistants (Anthropic Claude models) to read our 2025 training notebooks and reconstruct the history of the runs, to write the evaluation and analysis scripts, to search for and check references, and to draft parts of the text. The authors designed and ran the training in 2025, checked the reconstructed history and every number against the recorded files, and take full responsibility for the content. Every number in this report is generated by a script from logged files, and each reference was checked against its arXiv, ACL Anthology or DOI record. AI assistants are not authors.

## References

Mullosharaf K. Arabov. 2026. Benchmarking Parameter-Efficient Fine-Tuning of Large Language Models for Low-Resource Tajik Text Generation with the Tajik Web Corpus. Preprint, arXiv:2605.03742.

Mullosharif K. Arabov, Karomatullo Habibullozoda, and Nurali Shirinov. 2026. TajikNLP: An Open-Source Toolkit for Comprehensive Text Processing of Tajik (Cyrillic Script). Preprint, arXiv:2605.04583.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 749–775, Bangkok, Thailand. Association for Computational Linguistics.

Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John P. Cunningham. 2024. LoRA Learns Less and Forgets Less. Preprint, arXiv:2405.09673.

Tyler A. Chang, Catherine Arnett, Zhuowen Tu, and Benjamin K. Bergen. 2023. When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages. Preprint, arXiv:2311.09205.

Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca. Preprint, arXiv:2304.08177.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient Finetuning of Quantized LLMs. Preprint, arXiv:2305.14314.

Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L. Richter, Quentin Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. 2023. Continual pre-training of large language models: How to (re)warm your model? Preprint, arXiv:2308.04014.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR).

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. Preprint, arXiv:2106.09685.

Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, and Irina Rish. 2024. Simple and scalable strategies to continually pre-train large language models. Preprint, arXiv:2403.08763.

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A. Choquette-Choo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, and Orhan Firat. 2023. MADLAD-400: A Multilingual And Document-Level Large Audited Dataset. Preprint, arXiv:2309.04662.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2023. An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning. Preprint, arXiv:2308.08747.

Lorenz K. Müller, Philippe Bich, Jiawei Zhuang, et al. 2025. SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights. Preprint, arXiv:2509.22944.

Guilherme Penedo, Hynek Kydlícek, Vinko Sabolˇ cec,ˇ Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. 2025. FineWeb2: One Pipeline to Scale Them All – Adapting Pre-Training Data Processing to Every Language. Preprint, arXiv:2506.20920.

NLLB Team, Marta R. Costa-jussà, James Cross, et al. 2022. No Language Left Behind: Scaling Human-Centered Machine Translation. Preprint, arXiv:2207.04672.

An Yang, Anfeng Li, Baosong Yang, et al. 2025. Qwen3 Technical Report. Preprint, arXiv:2505.09388.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2023. Large Language Models Are Not Robust Multiple Choice Selectors. Preprint, arXiv:2309.03882.