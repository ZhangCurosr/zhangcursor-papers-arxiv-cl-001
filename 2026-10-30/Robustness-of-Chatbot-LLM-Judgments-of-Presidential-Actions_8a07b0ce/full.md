# Robustness of Chatbot LLM Judgments of Presidential Actions and Hypothetical Reforms.

Oleg Vasilyev Independent Researcher vasilyev.oleg@gmail.com

## Abstract

Chatbots are increasingly consulted about public affairs, and their judgments of policies are useful only if they are robust. We measure this robustness in the setting of an ordinary user, through consumer chat interfaces. For 30 days, five chatbot configurations (Mistral, Gemini Flash, Claude Sonnet, and Claude Opus at two effort settings) were asked twice a day to score the long-term effect on the United States of 40 actions of the last four presidential administrations and of 15 hypothetical reforms, each with a confidence and a short comment. The judgments are robust: all configurations agree on the sign and the ranking of the administrations’ mean scores, and two different configurations asked on the same day give opposite signs to the same action in about 4% of cases. Responses from different days differ more than same-day responses, but the difference grows slowly over the month, except where the service itself changed. We identify three such changes, which shifted the judgments while leaving little or no trace in the interface. Treating the comments as a retrieval task, we find them specific to each item and consistent across runs and chatbots, with lexical and embedding similarities agreeing. We release the dataset of 16,500 judgments.

## 1 Introduction

Chatbots are becoming a common source of information on public affairs (Reuters Institute for the Study of Journalism, 2026; Bruinsma et al., 2026). AI systems have also been shown to help people find common ground in political deliberation (Tessler et al., 2024; Peters et al., 2026). Judging high-level human decisions with AI may be helpful for two reasons:

1. The judgment is mediated by a model and by the accumulated human knowledge it has learned, rather than driven by immediate political stimuli and the emotional impulses of a specific person or a small group of people.

2. The judgment is fast and can cover many factors and possible consequences.

Such judgments are useful only if they are robust. It is therefore important to estimate, at least empirically and crudely, how robust AI judgments of complex human decisions are. We consider not an evaluation system built around fixed versions of specific models, but the practical setting of a person using a chatbot, who controls neither the model version nor the routing, system, tools or retrieval behind the interface. The practical questions are:

1. How robust is a judgment to the choice of chatbot?

2. How robust is it to drift from day to day, as knowledge and the systems change?

3. How robust is it within a day, given the randomness that sampling, routing and tool use bring in?

4. Does robustness differ between judging decisions already implemented and judging hypothetical ones?

Political views of LLMs. Most studies of LLMs and politics measure a model’s own positions with opinion surveys and political questionnaires, and typically find left-of-center leanings (Santurkar et al., 2023; Rozado, 2024). Such studies compare a model’s opinions with those of a population or of its subgroups, but which group, if any, a model should agree with is unclear, as Santurkar et al. (2023) themselves note. Moreover, the robustness of such measurements has been questioned: positions elicited with forced-choice questions change when models may respond freely (Röttger et al., 2024), and the abstract positions of models do not predict how they vote on concrete referenda (Barmettler, 2026). On the other hand, responses are fairly consistent across paraphrases, formats and languages, though less so on controversial topics (Moore et al., 2024), and stable across repeated runs of the same questionnaire (Sakhawat et al.,

2026; Hanna and Hanna, 2026). Here we ask not for agreement with statements but for quantified judgments of the consequences of specific government actions, real and hypothetical.

Change over time. Chen et al. (2024) showed that the behavior of GPT-3.5 and GPT-4 changed substantially between their March and June 2023 versions. Cen et al. (2025) queried twelve models almost daily during the 2024 US election season and found both gradual drift and abrupt steps, some coinciding with known model updates and others unexplained. Over 36 weeks of weekly queries, Nogueira et al. (2026) found a small drift to the right in the political responses of ChatGPT, Gemini and Claude, often coinciding with new releases, together with more hedging over time; Aksoy et al. (2026) propose a framework for tracking such biases across model releases.

Run-to-run variation and chat interfaces. Even at temperature zero, LLMs served through APIs give different outputs when a query is repeated, with accuracy varying by up to 15% between runs (Atıl et al., 2025). A chat interface adds further unknowns, since the model version, system prompt, routing and retrieval are chosen by the provider and may change without notice. In an audit of three chatbots answering prompts based on disinformation about the war in Ukraine through their standard interfaces, the same prompt submitted in new chats gave responses of different accuracy for up to 53% of the prompts (Makhortykh et al., 2024).

LLMs as judges. LLMs are widely used to judge the quality of texts, where their verdicts agree well with human preferences but show position, verbosity and self-enhancement biases (Zheng et al., 2023). In our case the object of judgment is a real or hypothetical government decision, and no reference response exists, so robustness is the property that can be measured.

This work. To answer the questions above in the setting of an ordinary user, we queried five chatbot configurations from three providers through their consumer chat interfaces every day for 30 days, from September 13 to October 12, 2026, in the weeks before the US midterm elections. Each day, each chatbot judged 40 real US presidential actions, from the Obama administration to the second Trump administration, and 15 hypothetical US government reforms, giving for each item a score, a confidence and a short comment. Each prompt was run twice a day, each time in a fresh temporary chat, the second time immediately after the first, which gives the same-day difference. Figure 1 illustrates all 600 prompt runs; together they contain 16,500 judgments.

![](images/a82cc134146c2740edae4e1c711db1bd77cf621b2003d0771eecdb6c76236f44.jpg)  
Figure 1: All 600 prompt runs: 5 chatbot configurations × 2 prompts × 30 days × 2 runs a day, shown as small boxes. The setup is explained in Section 2.

Our contributions are:

1. A public dataset of 16,500 judgments of 40 presidential actions and 15 hypothetical reforms by five chatbot configurations, collected twice a day over 30 days, each with a score, a confidence and a comment explaining the score<sup>1</sup>.

2. Evidence that the judgments are robust to the choice of chatbot and the day: all configurations agree on the sign of each administration’s mean score and on the ranking of the administrations; two different configurations asked on the same day give opposite signs to the same action in only about 4% of cases.

3. A comparison of the same-day noise, the dayto-day drift and the differences between chatbots; the identification of three changes in the services behind the chat interfaces during the month (the changes shifted the judgments while leaving little or no trace in the interface).

4. A use of standard retrieval measures to evaluate the consistency of the comments that explain the judgments. The comments are specific to each item and consistent across runs and chatbots; the conclusions agree across lexical and embedding similarity and across ranking measures. For the hypothetical reforms, word overlap identifies the comments across chatbots better than embeddings do, suggesting that specific terminology carries much of the distinction.

![](images/4c39e580ea4c3ef3a513c34d938d5ce87d354d54e42c0f2c0df0101067747197.jpg)  
Figure 2: The 40 presidential actions by the year in which each was launched (the first year, for actions that span several years). By administration: Obama 10, Trump1 9, Biden 9, Trump2 12.

## 2 Setup

## 2.1 Items to Judge

The presidential actions were selected with the help of Claude Opus 5, which we asked for the 40 most consequential presidential actions of the four latest US administrations, from the Obama administration to the beginning of 2026. Ending the selection there kept most of the actions within the training data of the models, and left more than half a year between the latest action (January 2026) and the start of the experiment.

The actions are listed with their descriptions in Table 6 (Appendix A.1), and their distribution over years and administrations is shown in Figure 2.

For more challenging judgments, we also defined, with the selection guided by Claude Opus 5, 15 hypothetical US government reforms intended to be among the most beneficial that could be proposed (Table 7, Appendix A.2). Six of them reform political institutions and state capacity (I01– I06), eight are economic and social policies, from housing and permitting to education and the fiscal path (I07–I14), and one insures against catastrophic risks, from pandemics to AI (I15).

Some of the reforms are arguably too hypothetical: they are designed to remedy the very conditions that prevent such reforms from being launched. If the government could be motivated to launch them, there might be no need for them. The exercise is therefore a thought experiment: we assume that, for a short interval of time, the motivation of the politicians switches to the benefit of the country, and that the reforms are launched within that interval. The prompt expresses this by asking the models to assume that each reform is taken now and fully supported by all branches of government.

## 2.2 Judgment

The judgment of the actions is defined by the prompt in Figure 12. It asks for the net effect of each action on the long-term flourishing of the US population over the 50 years after the action was taken. The score is an integer from −5 to +5: its sign gives the direction of the effect, and its size is anchored to rough numbers of lives, shares of national income, and changes to core institutions. The confidence, an integer from 1 to 5, states how sure the model is of the score, including its sign, and a comment of at most 100 characters gives the reason. The models are asked to judge each action on its own merits, independently of the other actions and of any overall view of the administration, to include consequences that became apparent later, and to base the score on empirical evidence rather than on a political judgment.

The judgment of the reforms is defined by the prompt in Figure 13. It asks for the effect of each reform on human flourishing in the United States over the next 50 years, compared with a baseline of current law and current trends, in expectation rather than as the most likely outcome, and ignoring implementation cost and political difficulty. The score (called magnitude in the prompt and in the dataset) is an integer from 1 to 5 and has no sign, because the reforms were designed to be beneficial; a reform judged harmful falls in the lowest band. The bands are anchored to the permanent change in national income, with averted deaths converted at roughly \$10M each, and to the scale of institutional change. Besides the confidence and the comment, the prompt asks for:

1. Dimension: the dimension that determined the score: lives, income or institutional.

2. Irreversibility (1 to 5): how hard the reform would be to undo within 50 years if it proved wrong; 5 means constitutional or effectively permanent, 1 repealable by ordinary legislation.

3. Contestedness (1 to 5): how much informed experts disagree about the score, independently of the model’s own confidence; 5 means deeply contested, 1 not contested.

Both prompts ask only for the effect on the United States, ignoring the rest of the world, because this is common in US policy debate, and it makes the scoring easier. Both prompts also require outpus as one line per item in a fixed format, which was parsed into the fields of the dataset.

## 2.3 Chatbots

Each day from September 13 to October 12, 2026, both prompts were run twice on each of the following chatbot configurations (the short notations are used in the figures):

1. M: Mistral’s Vibe, Think mode (see Section 3.5.2);

2. F: Gemini Flash 3.6;

3. S: Claude Sonnet 5, effort High;

4. Oh: Claude Opus 5, effort High;

5. Om: Claude Opus 5, effort Max.

Each run was made in a fresh temporary chat, so that no earlier conversation could influence the responses, and the items were listed in a random order that differed between the runs. The second run of the day followed immediately after the first. When a chatbot refused to score or broke the output format, the run was repeated in a new chat.

Throughout the paper we call these chatbot configurations models for simplicity, and number the days from 1 (September 13) to 30 (October 12).

## 2.4 Dataset

The resulting dataset consists of four parts:

1. Presidential actions (40): id, administration, year, name, description.

2. Hypothetical reforms (15): id, name, description.

3. Scored actions (12,000 = 30 days × 5 models × 2 runs × 40 actions): day, date, model, run, item id, score, confidence, comment.

4. Scored reforms (4,500): day, date, model, run, item id, score, dimension, irreversibility, contestedness, confidence, comment.

It is described in Appendix B and published on Hugging Face.<sup>2</sup>

## 3 Observations

## 3.1 Scores by Administration

The simplest aggregation is by administration and model. Figure 3 shows each model’s mean score and mean confidence for the actions of each administration, averaged over all days and runs. The aggregates are consistent across the models:

1. The sign of the mean score is the same for all models: positive for the Obama and Biden administrations, and negative for the two Trump administrations.

![](images/7fcfc887d851ab2b185752fbc76fe3ab7f7283bf0f56a90896e0d0d223e205e2.jpg)

![](images/cea6132673dbf65cbec68e225c018b93484f1598f9b1f5ad9a7798f7f8019268.jpg)

<table><tr><td rowspan=1 colspan=4>Mean confidence</td></tr><tr><td rowspan=1 colspan=1>3.8</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>3.0</td></tr><tr><td rowspan=1 colspan=1>4.1</td><td rowspan=1 colspan=1>3.9</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>3.5</td></tr><tr><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.4</td></tr><tr><td rowspan=1 colspan=1>3.5</td><td rowspan=1 colspan=1>3.1</td><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>2.9</td></tr><tr><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>2.9</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=2 colspan=3>3.0 3.5 4.0 4.5</td></tr><tr><td rowspan=1 colspan=1>2.0 2.5</td></tr></table>

Figure 3: Mean score (left) and mean confidence (right) of each model for the actions of each administration, averaged over all days and runs. O: Obama; T1 and T2: the first and second Trump administrations; B: Biden.

2. All models rank the administrations in the same order: Obama, Biden, the first Trump administration, the second.

3. The mean confidence is moderate to high for the Obama actions (3.0 to 4.1), the earliest ones, about which the most evidence has accumulated, and falls toward moderate for the later administrations (2.4 to 3.5 for the second Trump administration). The confidence in the judgments of the hypothetical reforms (2.4 to 3.3) is only a little lower than for the actions of the second Trump administration.

Figure 4 shows the same data more concisely. The ranges of the five models’ mean scores do not overlap between the administrations; averaged over the models, the mean score is +2.65 for Obama, +2.04 for Biden, −0.67 for the first Trump administration and −2.95 for the second. We do not discuss here why the models judge the actions of the Trump administrations, especially the second, so negatively: on the scale of the prompt, a score of −3 means a large and durable negative effect. The decline may reflect what Kemp (2025) calls the Goliath’s Curse. For our purposes it is enough that very different models agree in this scoring.

## 3.2 Scores of Individual Items

Aggregating only over the models, days and runs, Figure 5 shows the distribution of the 300 scores on each item (5 models × 30 days × 2 runs). The scores cluster tightly, although they come from five different models and from 30 days. On average, 94% of the scores on an action, and 97% of the scores on a reform, lie within one point of the item’s median. The sign of an action’s score agrees with the sign of its median in 95% of the responses. The most divided actions are the reshaping of the federal judiciary (I20), whose scores span the whole scale from −5 to +5, and the expanded parole and asylum admissions (I35), scored positive in 54% of the responses, negative in 29% and zero in 17%. Figure 14 (Appendix C) shows the distributions for each model separately.

![](images/3878643eb458b0488de730c63daa733e0efaf971c77a28c56fe0f808e56660d7.jpg)  
Figure 4: Summary of Figure 3. Left: for each administration, the bar spans the lowest and the highest of the five models’ mean scores, and the dark line is their average. Right: the same for the confidence, with the confidence in the hypothetical reforms (R) added on the right.

## 3.3 Difference between Models

Let $v _ { M } ( x , t , i )$ be a numeric field of the response of model M on item x (an action or a reform), on day t, in run $i = 1 , 2 \colon$ the score, the confidence, or, for the reforms, also the irreversibility or the contestedness. We define the difference between two models $M _ { 1 } \ne M _ { 2 }$ as

$$
\frac { 1 } { 4 | X | | T | } \sum _ { x , t , i , j } \left| v _ { M _ { 1 } } ( x , t , i ) - v _ { M _ { 2 } } ( x , t , j ) \right|\tag{1}
$$

Here $| X |$ is the number of the items (40 actions or 15 reforms), $| T | = 3 0$ is the number of the days, and the runs $i , j = 1 , 2 ;$ only the responses of the same day are compared.

The model’s own noise is the difference between its two runs of the same day:

$$
\frac { 1 } { 2 | X | | T | } \sum _ { x , t , i \neq j } \left| v _ { M } ( x , t , i ) - v _ { M } ( x , t , j ) \right|\tag{2}
$$

Figure 6 shows both for the scores, with Equation 1 off the diagonal and Equation 2 on it. Two runs of the same model on the same day differ by 0.27 to 0.39 points on the action scale for the

Claude models and Gemini, and by 0.55 for Mistral. Different models differ by 0.67 to 0.88 points, with one exception: the two Opus settings differ by only 0.38, about as much as each differs from itself. The reform scores show the same structure on their scale from 1 to 5: 0.21 to 0.43 on the diagonal, and 0.41 to 0.86 off it.

Figure 7 shows the same for the other fields. For every field, including the scores, the three Claude configurations are closer to each other than to Gemini or Mistral. By administration (Figure 15, Appendix D), the models differ least on the Obama actions and most on the actions of the first Trump administration.

Figure 8 shows how often two responses on the same action disagree more strongly: in sign, or by more than one point. Encouragingly, two different models asked about the same action on the same day give opposite signs in only 4.3% of the pairs of responses on average (from 1.3% to 7.3% for individual pairs of models), and differ by more than one point in 12%. Between the two runs of the same model, opposite signs occur in less than 1% of the pairs for the Claude models, in 2.3% for Mistral, and in almost 4% for Gemini, which also disagrees most with the other models.

## 3.4 Longitudinal Drift

The responses may also drift from day to day. Possible reasons include changes in the routing of a chatbot, updates of the model, and shifts in public knowledge and sentiment, which the chatbots can reach through web search. The period of the experiment, the weeks before the November 2026 midterm elections, may amplify such drift.

For a model M, an item x and a number of days $d > 0$ , we define the drift as

$$
\frac { 1 } { 4 | T _ { d } | } \sum _ { t , i , j } \bigl | v _ { M } ( x , t + d , i ) - v _ { M } ( x , t , j ) \bigr |\tag{3}
$$

Here $T _ { d }$ is the set of the days t for which $t + d$ is also a day of the experiment $( | T _ { d } | = 3 0 - d )$ and the other notations are as in Equation 1. For $d = 0$ , only the different runs of the same day are compared:

$$
\frac { 1 } { 2 | T _ { 0 } | } \sum _ { t , i \neq j } \left| v _ { M } ( x , t , i ) - v _ { M } ( x , t , j ) \right|\tag{4}
$$

which is Equation 2 for a single item. We use d from 0 to 25, because fewer than five pairs of days are left for larger d.

![](images/2404381f5382d0e25ae473b8fa4ba8a1cfa467674bd3515d9e5871d6e9b595c7.jpg)

![](images/3a845f56a7b74c9d19769c56d2d7c248c47b42290c19f9ebad80c21b9cbb356f.jpg)  
Figure 5: Distribution of the scores of each item over all models, days and runs (300 responses per item). The area of a dot is proportional to the share of the responses. The actions are in chronological order and colored by administration; the reforms are in the order of their ids. The score scales are defined in Figures 12 and 13.  
Figure 6: Mean absolute difference between the scores of two models on the same item and day (Equation 1, off the diagonal), and between the two runs of the same model (Equation 2, on the diagonal), for the actions (left) and the reforms (right).

Figure 9 shows the drift of the scores, averaged over the items. For every model, the responses of consecutive days (d = 1) differ more than the two runs of the same day (d = 0); for the actions, by 0.08 to 0.20 points. After the first day, the drift grows slowly: from d = 1 to d = 25, by 0.06 to 0.14 points for the actions, and by less, or not at all, for the reforms. The exception is Mistral, whose drift for the actions keeps rising, from 0.67 to about 1.0, mostly because of a single change of its system in the middle of the month (Section 3.5.2).

For the actions, a model’s responses 25 days apart remain closer to each other than to another model’s responses on the same day: after 25 days, the drift of Gemini, Sonnet and the two Opus settings is 0.46 to 0.62, while their same-day differences from the other models are 0.67 to 0.88 (Figure 6). Waiting a month thus changes a judgment less than switching to another chatbot. The exceptions are the two Opus settings, which differ from each other by only 0.38, and Mistral, whose drift reaches 1.0 because of its change of system.

The drift is not the same for all items. For each model, Figures 16–20 in Appendix E show the median over the items and the range between their first and third quartiles. The median is somewhat below the mean, which is raised by a minority of items that drift more: after 25 days, a quarter of the actions drift by more than 0.6 to 0.7 points (1.15 for Mistral).

## 3.5 Changes in the Models

Figure 21 (Appendix F) shows the daily means of the scores and of the other fields, averaged over the items and the two runs of each day. Besides the day-to-day noise, it shows changes that last for several days. At least three of them coincide with identifiable events and can be confirmed statistically:

1. Claude Sonnet 5: on several days it reported its knowledge cutoff as August 2026 instead of January 2026, and its responses changed on exactly those days (Section 3.5.1).

2. Mistral: its chat interface changed in the middle of the month, and its responses changed with it for the rest of the month (Section 3.5.2).

3. Claude Opus 5: on five days in the week before the release of Opus 5.5, both effort settings responded markedly more neutrally (Section 3.5.3).

![](images/3d968097e31ade52b9290f39c97eb70cc242ecdcf082cf9b6155b2957446a6e2.jpg)

![](images/e31a3bb65b7a6a71cb9d076e1ae99bb7f7cc8f6d434bd3aadab5ac06dfdd732f.jpg)  
Figure 7: As Figure 6, for the confidence of the actions and for the confidence, contestedness and irreversibility of the reforms.

![](images/e59108c91239dc0c8113f74c88f4ded52734c493c903a09df90f7e9e8d0861e4.jpg)

![](images/d462462f1be44a13b7f2f5632ac06b52ab58f66d4e9644aa22f0f06887052d8b.jpg)

Figure 8: Strong disagreements between responses on the same action on the same day: the share of the pairs of responses with opposite score signs (left), and with scores that differ by more than 1 (right). The pairs are formed as in Equations 1 and 2; on the diagonal, the two runs of the same model.
<table><tr><td></td><td>August</td><td>January</td><td>p</td></tr><tr><td>Comment length</td><td>82.5</td><td>71.0</td><td>&lt;0.0001</td></tr><tr><td>Reforms score</td><td>3.41</td><td>3.67</td><td>0.0004</td></tr><tr><td>Reform contestedness</td><td>3.49</td><td>3.78</td><td>&lt;0.0001</td></tr><tr><td>Share of action score = 0</td><td>4.6%</td><td>2.1%</td><td>0.0203</td></tr></table>

Table 1: Sonnet’s answers on the days it reported a different knowledge cutoff. August: the 6 days (6–9, 11, 12) on which Sonnet, asked daily for its knowledge cutoff, answered “End of August 2026”; January: the 23 days on which it answered “January 2026”. Day 16, when it gave both answers, is left out. Each quantity is averaged per day over items and both runs, then over the days of each group. Comment length is in characters and pools the comments on actions and reforms; reforms score is the magnitude (1–5). p: exact two-sided permutation test on the daily values, over all 475,020 ways of choosing 6 of the 29 days.

## 3.5.1 Sonnet 5

Each day we also asked the Claude models for their knowledge cutoff and the current date. Sonnet 5 usually answered “January 2026” or “End of

January $2 0 2 6 ^ { \circ }$ , but on six days (18–21 and 23–24 September; days 6–9, 11 and 12) it answered “End of August 2026”; on 28 September (day 16) it gave both answers at different times of the day. Table 1 shows that its responses to our prompts changed on exactly those days. Its comments were about 16% longer, and it rated the reforms both as having smaller effects and as less contested; it also gave a score of zero to actions about twice as often. Its scores of the actions, by contrast, changed little: neither their size nor its confidence differed significantly between the two groups of days. The self-reported cutoff is not a reliable identifier of a model, but here it varies together with how the model writes and judges, which suggests that a different model or configuration was served under the same name on those days. It was not the released Sonnet 5.5: when that model appeared in the interface on 28 September, it reported a cutoff of June 2026.

Figure 10 shows two of these quantities day by day. Comment length varies from day to day on its own (it is also high on days 15, 25 and 29), so the evidence lies in the clustering of high values on the shaded days rather than in any single day. The clearest sign is day 10, the one day inside this period on which Sonnet 5 again reported “January 2026”: both comment length and contestedness return to their usual level on that day, and depart from it again on day 11. A user of the chat interface had no way to notice this change except by asking the model about itself.

## 3.5.2 Mistral

On 27 September (day 15) the Mistral interface we used changed: the incognito mode was renamed “Temporary chat”, and on the following day the separate Think option disappeared. This was our account’s migration to Mistral’s new experience, which merged the earlier Chat and Work interfaces; according to the provider’s release notes, free and paid accounts were migrated in phases starting 14 September (Mistral AI, 2026a). Until then our Mistral condition used the Think mode of the legacy Chat, documented as powered by Mistral’s Magistral reasoning models (Mistral AI, 2026b). In the new experience, Think is a toggle that switches to “deeper, agentic workflows”, and the model behind it is not stated (Mistral AI, 2026c). Our prompts did not change.

![](images/9b4fa45b6a1fab36efea5de60eb8f2baff8de65c8721d3e348c5b46254864ac5.jpg)  
Figure 9: Drift of the scores (Equations 3 and 4), averaged over the items, as a function of the number of days d between the responses.

![](images/2d4197db858b799144e35245c6676dada01bacf9c489a33da2fc2a406fabfd85.jpg)  
Figure 10: Sonnet’s daily mean comment length (left) and mean contestedness of the reforms (right). Shaded: the days on which Sonnet reported its knowledge cutoff as “End of August 2026”; lighter: day 16, when it gave both responses. Dashed: the mean over the days on which it reported “January 2026”.

Table 2 compares Mistral’s responses in the 14 days before and the 16 days after the switch. All eight quantities changed. The comments became three to four times longer, confidence fell by about one point on both actions and reforms, and the scores became milder. The new system was also more consistent: its two daily runs differed less from each other, and its action scores moved much closer to the mean of the other four models, a distance that fell by 40%. For the reforms, that distance did not change (0.75 in both periods). For five of the eight quantities the two periods do not overlap at all: every day after the switch lies beyond every day before it.

<table><tr><td></td><td>Before</td><td>After</td><td>p</td></tr><tr><td>Comment length, actions</td><td>30.8</td><td>89.5</td><td>&lt;0.0001 †</td></tr><tr><td>Comment length, reforms</td><td>22.6</td><td>86.5</td><td>&lt;0.0001†</td></tr><tr><td>Action |score|</td><td>3.20</td><td>2.50</td><td>&lt;0.0001</td></tr><tr><td>Action confidence</td><td>4.05</td><td>2.92</td><td>&lt;0.0001 †</td></tr><tr><td>Reforms score</td><td>3.80</td><td>3.16</td><td>&lt;0.0001</td></tr><tr><td>Reforms confidence</td><td>3.74</td><td>2.72</td><td>&lt;0.0001</td></tr><tr><td>Run-to-run difference</td><td>0.66</td><td>0.46</td><td>0.0031</td></tr><tr><td>Distance to other models</td><td>0.85</td><td>0.51</td><td>&lt;0.0001 †</td></tr></table>

Table 2: Mistral before (13–26 September, days 1–14) and after (27 September–12 October, days 15–30) its switch to the new Vibe experience. Each quantity is computed per day over items and both runs, then averaged over the days of each period. Comment length is in characters. Reforms score is the magnitude (1–5). Runto-run difference: mean absolute difference between the two daily runs’ action scores (Eq. 2 for Mistral alone). Distance to other models: mean absolute difference between Mistral’s action score and the mean score of the other four models on the same day. p: two-sided permutation test on the daily values, $1 0 ^ { \overset { \cdot } { 6 } }$ random splits of the 30 days. <sup>†</sup>Every day after the switch lies beyond every day before it.

Figure 21 shows that the change was a single step rather than a gradual drift. Mistral’s daily confidence and scores fall on day 15 and stay at the new level for the rest of the month. The steady rise of Mistral’s drift curves in Figure 9 therefore reflects mostly this one step, and any analysis that assumes a fixed system should treat the Mistral data as coming from two systems, days 1–14 and days 15–30. A user asking the same chatbot the same question would have received systematically different, and more cautious, judgments after a change that showed up in the interface mainly as a renamed option.

<table><tr><td></td><td colspan="2">Before 22 Sep</td><td colspan="2">From</td></tr><tr><td></td><td>Unusual</td><td>Other</td><td>22 Sep</td><td>p</td></tr><tr><td>Opus high</td></tr><tr><td>Action |score|</td><td>2.04 2.80</td><td>2.81</td><td>0.0004</td></tr><tr><td>Action confidence</td><td>2.68</td><td>3.16</td><td>3.18</td><td>0.0005</td></tr><tr><td>Share of score = 0</td><td>8.0%</td><td>0.3%</td><td>1.0%</td><td>0.0019</td></tr><tr><td>Hedging comments</td><td>12.5%</td><td>6.2%</td><td>7.0%</td><td>0.0242</td></tr><tr><td>Reforms score</td><td>3.47</td><td>3.80</td><td>3.77</td><td>0.0553</td></tr><tr><td>Reforms confidence</td><td>2.29</td><td>2.88</td><td>2.85</td><td>0.0185</td></tr><tr><td colspan="5">Opus max</td></tr><tr><td>Action |score|</td><td>1.97</td><td>2.89</td><td>2.86</td><td>0.0007</td></tr><tr><td>Action confidence</td><td>2.56</td><td>3.00</td><td>3.06</td><td>&lt;0.0001</td></tr><tr><td>Share of score = 0</td><td>14.3%</td><td>2.5%</td><td>0.2%</td><td>&lt;0.0001</td></tr><tr><td>Hedging comments</td><td>12.5%</td><td>6.2%</td><td>4.8%</td><td>0.0005</td></tr><tr><td>Reforms score</td><td>3.08</td><td>3.85</td><td>3.86</td><td>0.0005</td></tr><tr><td>Reforms confidence</td><td>2.19</td><td>2.67</td><td>2.69</td><td>0.0004</td></tr></table>

Table 3: Opus 5 in the week before Opus 5.5 was released. Each quantity is computed per day over items and both runs, then averaged over three groups of days: Unusual, the days 3, 4, 6, 7 and 9 (15–21 September); Other, the other days before the release (1, 2, 5, 8); and the days from the release of Opus 5.5 on 22 September on (10–30). Share of score = 0 and hedging comments refer to the actions; a comment counts as hedging if it contains “uncertain”, “unclear”, “contested”, “disputed”, “debat(e)”, “mixed”, “too recent” or “too early”. Reforms score is the magnitude (1–5). The unusual days were identified in the data and are shown for description; p tests the split fixed by the release date, all days before it (1–9) against all days from it (10–30): two-sided permutation test on the daily values, 10<sup>6</sup> random splits.

## 3.5.3 Opus 5

On five of the seven days from 15 to 21 September (days 3, 4, 6, 7 and 9), both Opus settings responded markedly more neutrally (Table 3). Their scores moved toward zero for all four administrations. The share of zero scores (“no material longrun effect”), otherwise between 0.2% and 2.5%, rose to 8% for Opus high and 14% for Opus max, mostly for actions of the first Trump and the Biden administrations; comments containing words of doubt such as “uncertain” or “contested” became about twice as frequent; and confidence fell by about half a point on both actions and reforms. Other signs stayed the same: the comments were as long as on other days, and asked for its knowledge cutoff, Opus answered “end of May 2026” every day of the experiment. The two settings moved together (the correlation of their daily mean absolute score over the 30 days is 0.93), so the cause lay in what was served under both rather than in the effort setting.

The episode ended on 22 September (day 10), the day Opus 5.5 was released (Anthropic, 2026b) and appeared in the model selector. We continued to use Opus 5, which from that day responded at its usual level for the rest of the month. Anthropic’s published system prompt for Opus 5 shows a single version, dated 24 July 2026 (Anthropic, 2026a), so no documented change to it falls within the experiment, and we do not know what caused the episode. Because the five days were identified in the data, Table 3 reports p-values for a split fixed by the release date instead, all days before it against all days from it. Even this diluted comparison, which counts four ordinary days in the first group, is significant at the 5% level for 11 of the 12 quantities. Unlike Sonnet’s change, this one left no trace in the model’s responses about itself or in the style of its comments. It was visible only in its judgments.

## 3.6 Reasoning

## 3.6.1 Leading Dimension of the Reforms

The models agreed almost completely on which dimension determined the magnitude of a reform. All the models, on almost every day and in almost every run, assigned the reforms I01–I06 (political institutions and state capacity) to institutional, the reforms I07–I14 to income, and I15 (insurance against catastrophes) to lives.

Only 51 of the 4,500 responses (1.1%) chose a different dimension: 22 of Mistral, 18 of Opus high, 5 of Gemini, 4 of Sonnet and 2 of Opus max. The departures concentrate on the reforms whose effects plausibly fall on more than one dimension:

1. Early childhood (I11) as lives instead of income: 17.

2. State capacity (I06) as income instead of institutional: 14.

3. Catastrophe insurance (I15) as income instead of lives: 7.

4. Fiscal path (I14) as institutional: 5.

5. Reading and math (I12) as lives: 5.

The departures show no pattern over time: they are scattered across the month, and Mistral’s switch on day 15 made no difference (9 departures before it, 13 after).

## 3.6.2 Consistency of Comments

As requested by the prompts (Figures 12 and 13), each response contains a short comment (11 words in the median) giving the reason for the score. To ask whether two models give the same kinds of reasons, we measure how well the comments of one model on an item identify the comments of the other model on the same item among all its comments. This lets us use standard retrieval measures to evaluate the consistency of the reasons.

Let $c ( M , x , t , i )$ be the comment of model M on item x on day t in run i, and $C ( M , x , i )$ the set of its 30 comments on x in run i:

$$
C ( M , x , i ) = \{ c ( M , x , t , i ) : t \in T \}\tag{5}
$$

For the similarity $S ( c , C )$ between a single comment c and a set of comments $C _ { i }$ , we use the Jensen– Shannon divergence or embedding models.

With the Jensen–Shannon divergence:

$$
S _ { J S } ( c , C ) = 1 - \mathrm { J S } \big ( p ( c ) , { \frac { 1 } { | C | } } \sum _ { c ^ { \prime } \in C } p ( c ^ { \prime } ) \big )\tag{6}
$$

where $p ( c )$ is the distribution of the content words of $c$ (lower-cased words, without a short list of function words and without stemming), and JS is the Jensen–Shannon divergence in bits (Lin, 1991), which has long been used to compare the word distributions of texts, for example in the evaluation of summaries without human references (Louis and Nenkova, 2009). Each comment in C thus has equal weight, whatever its length.

With an embedding model e:

$$
\begin{array} { r } { S _ { e } ( c , C ) = \cos \big ( e ( c ) , ~ \frac { 1 } { | C | } \sum _ { c ^ { \prime } \in C } e ( c ^ { \prime } ) \big ) } \end{array}\tag{7}
$$

where $e ( c )$ is the unit-length embedding of c.

We use three embedding models of different architectures:

1. gina: jina-embeddings-v5-text-small, a decoder built on Qwen3 (Jina AI, 2026).

2. harrier: harrier-oss-v1-270m, a decoder built on Gemma 3 (Microsoft, 2026).

3. granite: granite-embedding-97mmultilingual-r2, a ModernBERT encoder (IBM, 2026).

Each model is used with the prompt or task setting that its model card specifies for text similarity.

For a query model $M _ { 1 }$ , an item x and a ranked model $M _ { 2 }$ , all $3 0 | X |$ comments of $M _ { 2 }$ in run 2 are ranked by $S \big ( c ( M _ { 2 } , x ^ { \prime } , t , 2 ) , C ( M _ { 1 } , x , 1 ) \big )$ , and the

30 comments with $x ^ { \prime } = x$ are relevant. We summarize each ranking by its average precision and by its R-precision, the share of relevant comments among the first 30 (Manning et al., 2008), and average over the |X| items and over both assignments of the runs (query from run 1 and ranked comments from run 2, and the reverse). This gives one value for each of the 25 ordered pairs $( M _ { 1 } , M _ { 2 } )$ . On the diagonal, the query and the ranked comments come from different runs of the same model, so the diagonal is a run-to-run baseline. A random ranking gives about $1 / | X | \colon 0 . 0 2 5$ for the actions and 0.067 for the reforms.

Figure 11 shows the mean average precision (MAP). The comments are highly specific to the items: a model’s own comments from the other run are retrieved with MAP between 0.92 and 0.95 on average, depending on the similarity measure, and another model’s comments with 0.80–0.84 for the actions and 0.74–0.80 for the reforms, far above chance. Part of this success comes from topic words that every model uses for a given item, so the differences between pairs reflect how similarly the models phrase their reasons. The Claude models form a family: the two Opus settings are nearly interchangeable (0.92–0.97 for the actions, 0.84– 0.95 for the reforms), and Sonnet is close to both. Gemini and Mistral each comment in their own way. The weakest pairs, 0.54–0.66, always join Opus max with Mistral or Gemini. R-precision gives the same picture (Figure 22 in Appendix G).

These conclusions do not depend on the similarity measure. Across the four measures, the offdiagonal cells are ordered alike: Spearman correlations between measures range from 0.84 to 0.98, and from 0.93 to 0.98 among the three embedding models. Nor do they depend on the ranking metric (Table 4). R-precision, MAP and nDCG truncated at $K = 3 0$ , and full nDCG order the cells as MAP does $( \rho \ge 0 . 9 8 )$ . The reciprocal rank of the first relevant comment (MRR) is at its ceiling: the top-ranked comment is about the right item in almost every query, and its weak correlation with MAP reflects noise near 1. AUC follows MAP within a narrow range: a comment on the right item ranks above a comment on another item in 86.5% to 99.9% of all such pairs, for every pair of models.

The matrices are not symmetric, and the difference is informative (Table 5). Mistral’s averaged comments retrieve other models’ comments well (0.84 for the actions), but its own comments are the hardest to retrieve (0.73), and they are also the least consistent between its own two runs (0.88, against 0.96–0.97 for the other models). This fits the change of system on 27 September (Section 3.5.2), which mixes two styles of comments in Mistral’s data. Gemini shows the opposite pattern: its comments are the most consistent of all (0.96 for the actions, 0.98 for the reforms) and are found easily by other models’ queries (0.82), but its own averaged comment retrieves other models’ comments least well (0.75 and 0.70): Gemini phrases its reasons in a way of its own.

JS, reforms  
![](images/de82bb8c731e102d208e4eedd9263d6ea7f2f255e60a1d41f7fc368c08546ffe.jpg)

![](images/c96981b8735c2255c4c80fd0765f8986bc34af81ff0b7d724c408edc0c2447a7.jpg)

![](images/d37c02ef7464e758fdb627caad2a42df47de63525c52854e8757a74c45cd1df7.jpg)

![](images/1a0f0d75eb97f4bdc18708eda9196bf815329d48af3a6043c1fcaca8caef5f41.jpg)

![](images/310ab80e72fdf5fdc3c8b6c107e086ef82f67a3d1bf91143c0e0587ef433b1f3.jpg)

![](images/8a74b368be19649dc17c0ca49c3989d0fe53fec37df684e5040a2ea0c28f90d8.jpg)

![](images/28826d2c5593854315cf5155de40bee1936ad276507eb2dda8c1d375bc798886.jpg)

![](images/e84980e54b912b9c2f5df2ec7d6f5bbaa50fab9377e281ee4eb1bcfc1ab54ef0.jpg)  
Figure 11: Retrieval of comments across models: mean average precision for each query model (rows) and ranked model (columns), with Jensen–Shannon (JS) and three embedding models, for the actions (top) and the reforms (bottom). Diagonal cells (outlined): query and ranked comments from different runs of the same model. Chance level is about 0.025 for the actions and 0.067 for the reforms.

<table><tr><td>Metric</td><td>ρ</td><td> $\tau _ { b }$ </td><td>Range</td><td>≥ 0.99</td></tr><tr><td>MAP</td><td></td><td></td><td>0.54–0.97</td><td>0%</td></tr><tr><td>R-precision</td><td>0.98-1.00</td><td>0.88–0.98</td><td>0.49–0.93</td><td>0%</td></tr><tr><td>MAP@K</td><td>0.98-1.00</td><td>0.89-1.00</td><td>0.40-0.93</td><td>0%</td></tr><tr><td>nDCG@K</td><td>0.99-1.00</td><td>0.92–0.98</td><td>0.57–0.95</td><td>0%</td></tr><tr><td>nDCG</td><td>0.98–0.99</td><td>0.91-0.97</td><td>0.83-0.99</td><td>2%</td></tr><tr><td>AUC</td><td>0.93-0.97</td><td>0.79–0.89</td><td>0.86-1.00</td><td>22%</td></tr><tr><td>MRR</td><td>0.29–0.73</td><td>0.23-0.61</td><td>0.81-1.00</td><td>54%</td></tr></table>

Table 4: Agreement of other ranking metrics with MAP. For each similarity measure, and separately for the actions and the reforms, each metric gives a $5 \times 5$ matrix with one cell for each pair of a query model and a ranked model, as in Figure 11. Its 20 off-diagonal cells (pairs of different models) are compared with those of MAP, and the ranges are over the 8 matrices (4 similarity measures, for the actions and for the reforms). $\rho \colon$ Spearman rank correlation with MAP; $\tau _ { b } \colon$ Kendall’s tau-b with MAP, which accounts for ties (frequent for MRR). Range: smallest and largest off-diagonal value over all matrices; $\geq 0 . 9 9 \colon$ share of off-diagonal cells at 0.99 or above.

<table><tr><td></td><td colspan="3">Actions</td><td colspan="3">Reforms</td></tr><tr><td>Model</td><td>Own</td><td>Query</td><td>Ranked</td><td>Own</td><td>Query</td><td>Ranked</td></tr><tr><td>M</td><td>0.88</td><td>0.84</td><td>0.73</td><td>0.85</td><td>0.79</td><td>0.69</td></tr><tr><td>F</td><td>0.96</td><td>0.75</td><td>0.82</td><td>0.98</td><td>0.70</td><td>0.81</td></tr><tr><td>S</td><td>0.96</td><td>0.86</td><td>0.88</td><td>0.92</td><td>0.82</td><td>0.80</td></tr><tr><td>Oh</td><td>0.97</td><td>0.84</td><td>0.86</td><td>0.95</td><td>0.80</td><td>0.82</td></tr><tr><td>Om</td><td>0.96</td><td>0.79</td><td>0.80</td><td>0.93</td><td>0.75</td><td>0.72</td></tr></table>

Table 5: Each model as query and as ranked model: MAP averaged over the 4 similarity measures. Own: the diagonal cell, the model’s comments retrieved by its own query from the other run. Query: mean over the other four models of the model’s row, i.e. how well its averaged comments retrieve their comments. Ranked: mean of the model’s column, i.e. how well its comments are retrieved by the other models’ queries.

Word overlap and embeddings differ in one respect. With Jensen–Shannon, comments on reforms are retrieved across models as well as comments on actions (off-diagonal MAP 0.80 for both). With the embedding models they are retrieved less well (0.74–0.77 against 0.80–0.84). A plausible explanation is that the reforms are argued in a shared vocabulary of growth, institutions and long-run costs. Embeddings see comments on different reforms as close in meaning and confuse them more often, whereas word overlap remains anchored in the specific terms of each reform.

## 4 Conclusion

We asked five chatbot configurations, through their consumer chat interfaces, to judge 40 presidential actions and 15 hypothetical reforms twice a day for 30 days. Our experiment suggests that chatbots can give robust judgments of both real policies and hypothetical reforms: the scores, the characteristics that accompany them, and the comments that explain them distinguish the items well, and are reasonably stable against the choice of chatbot, the drift from day to day, and even the changes in the chatbot services. In terms of the questions of the introduction:

1. Choice of chatbot. The models agree on the sign of each administration’s mean score and on the ranking of the administrations, and two models asked about the same action on the same day give opposite signs in about 4% of the cases. The models differ more from each other than from themselves, and the three Claude configurations are closer to each other than to Gemini or Mistral, in their scores, in their other fields, and in the wording of their comments.

2. Drift from day to day. responses given on different days differ more than the two responses of the same day, but the difference grows slowly over the month. For the actions, a model’s responses given 25 days apart still differ less than its responses and those of another provider’s model given on the same day; the exception is Mistral, whose system changed in the middle of the month.

3. Variation within a day. Two runs of the same model on the same day differ by 0.27 to 0.55 points on the action scale, and give opposite signs in less than 1% of the cases for the Claude models and in almost 4% for Gemini.

4. Real and hypothetical decisions. The judgments of the hypothetical reforms behave much like those of the real actions: the responses cluster as closely around each item’s median, the models differ in the same pattern, and the drift over the month is, if anything, smaller. The reforms are, however, judged with lower confidence, about as low as the most recent actions, and the comments on different reforms are harder to tell apart by meaning than by wording.

The three changes that we identified show the main risk for a user: the system behind a chatbot’s name can change without notice, and shift its judgments for days or for good, while the interface shows little or nothing of it. A judgment that matters is therefore best checked with more than one chatbot, and on more than one day. The dataset that we release allows further analysis of these judgments, including the analysis of individual items, which we have left aside.

## Limitations

Our experiment and analysis have several limitations:

1. The interval of 30 days, although interesting because it falls in the heated weeks before the midterm elections, is short compared with the time scales of the presidential actions and the hypothetical reforms that we consider.

2. The set of models could be wider: we used five configurations from three providers, three of them from Anthropic.

3. Each set of items was judged with a single prompt, in English. We did not test how the judgments depend on the wording, format or language of the prompt, which can affect the responses of LLMs to political questions (Röttger et al., 2024; Moore et al., 2024), nor whether naming the administration of each action influences its score.

4. Both the actions and the reforms were selected with the help of Claude Opus 5, one of the models that we tested, and the selection may reflect its view of what is consequential.

5. Robustness is not correctness. Agreement between the models and stability over time do not show that the judgments are right; models trained on similar data may share the same errors. There is no reference response against which the judgments could be checked.

6. Chat interfaces give the user neither control over nor a record of the model version, system prompt and tools behind an response. The three changes that we describe were identified after the fact, from the data and from the models’ responses about themselves; smaller changes may have gone unnoticed.

7. We left out the analysis of the political or economic reasons for the scores themselves (Figures 4 and 5).

8. We did not analyze the individual items in depth: why particular actions or reforms are judged differently by different models, or how their judgments change over time.

9. The 50-year horizon of the prompts is a guess at a trade-off between a bias toward shortterm effects and the many unknowns of the distant future. Some discounting of later effects might be more appropriate, especially because large and unforeseeable changes are likely within such a horizon (Bostrom, 2019).

## References

Meltem Aksoy, Erik Weber, Jérôme Rutinowski, Niklas Jost, and Markus Pauly. 2026. Evaluating biases in large language models over time: A framework with a GPT case study on political bias. Applied Stochastic Models in Business and Industry, 42(2):e70078.

Anthropic. 2026a. Claude Opus 5 system prompts. ht tps://platform.claude.com/docs/en/rele ase-notes/system-prompts/claude-opus-5. Accessed 25 October 2026.

Anthropic. 2026b. Introducing Claude Opus 5.5. http s://www.anthropic.com/claude-opus-5-5. 22 September 2026. Accessed 25 October 2026.

Berk Atıl, Sarp Aykent, Alexa Chittams, Lisheng Fu, Rebecca J. Passonneau, Evan Radcliffe, Guru Rajan Rajagopal, Adam Sloan, Tomasz Tudrej, Ferhan Ture, Zhe Wu, Lixinyu Xu, and Breck Baldwin. 2025. Nondeterminism of “deterministic” LLM system settings in hosted environments. In Proceedings of the 5th Workshop on Evaluation and Comparison of NLP Systems, pages 135–148, Mumbai, India. Association for Computational Linguistics.

Joel P. Barmettler. 2026. Progressive in principle, centrist in practice: LLM political bias is instrumentdependent. arXiv, arXiv:2606.00048.

Nick Bostrom. 2019. The vulnerable world hypothesis. Global Policy, 10(4):455–476.

Bastiaan Bruinsma, Annika Fredén, Paul Röttger, Moa Johansson, and Asad Sayeed. 2026. Issue bias in generative AI writing assistance: Political issues and LLMs in the swedish 2026 election. arXiv, arXiv:2609.15207.

Sarah H. Cen, Andrew Ilyas, Hedi Driss, Charlotte Park, Aspen Hopkins, Chara Podimata, and Aleksander Madry. 2025. Large-scale, longitudinal study of large language models during the 2024 US election season. arXiv, arXiv:2509.18446.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. How is ChatGPT’s behavior changing over time? Harvard Data Science Review, 6(2).

Gabriel Hanna and Pierre Hanna. 2026. Stability and political orientation of international LLMs: An exploratory multi-run study conducted in French. arXiv, arXiv:2606.13760.

IBM. 2026. granite-embedding-97m-multilingual-r2. https://huggingface.co/ibm-granite/grani te-embedding-97m-multilingual-r2. Model card. Accessed 26 October 2026.

Jina AI. 2026. jina-embeddings-v5-text-small. https: //huggingface.co/jinaai/jina-embeddings-v 5-text-small. Model card. Accessed 26 October 2026.

Luke Kemp. 2025. Goliath’s Curse: The History and Future ofSocietal Collapse. Viking, London.

Jianhua Lin. 1991. Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1):145–151.

Annie Louis and Ani Nenkova. 2009. Automatically evaluating content selection in summarization without human models. In Proceedings ofthe 2009 Conference on Empirical Methods in Natural Language Processing, pages 306–314, Singapore. Association for Computational Linguistics.

Mykola Makhortykh, Maryna Sydorova, Ani Baghumyan, Victoria Vziatysheva, and Elizaveta Kuznetsova. 2024. Stochastic lies: How LLMpowered chatbots deal with Russian disinformation about the war in Ukraine. Harvard Kennedy School (HKS) Misinformation Review, 5(4).

Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. 2008. Introduction to Information Retrieval. Cambridge University Press.

Microsoft. 2026. harrier-oss-v1-270m. https://hugg ingface.co/microsoft/harrier-oss-v1-270m. Model card. Accessed 26 October 2026.

Mistral AI. 2026a. Release notes: A simpler Vibe experience. https://docs.mistral.ai/resource s/release-notes. Entry dated 22 September 2026. Accessed 24 October 2026.

Mistral AI. 2026b. Think mode. https://docs.mis tral.ai/vibe/chat-legacy/think-mode. Accessed 24 October 2026.

Mistral AI. 2026c. Vibe. https://docs.mistral.a i/vibe. Accessed 24 October 2026.

Jared Moore, Tanvi Deshpande, and Diyi Yang. 2024. Are large language models consistent over valueladen questions? In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15185–15221, Miami, Florida, USA. Association for Computational Linguistics.

Bruno Nogueira, Jost Große Perdekamp, Leon Swazinna, Elisabeth Kirsten, Nils C. Köbis, Juhi Kulshrestha, Markus Pauly, and Muhammad Bilal

Zafar. 2026. Changing tunes: A longitudinal study of political drift in LLMs. In ICML 2026 Workshop on Pluralistic Alignment.

Baran Peters, Gabor Hollbeck, Robert Jakob, and Kevin O’Sullivan. 2026. Polistemics: Evaluating LLMs as information mediators in politics & elections. arXiv, arXiv:2607.25953.

Reuters Institute for the Study of Journalism. 2026. Digital news report 2026. https://reutersinstitu te.politics.ox.ac.uk/digital-news-report/ 2026. Published 16 June 2026.

Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Kirk, Hinrich Schuetze, and Dirk Hovy. 2024. Political compass or spinning arrow? towards more meaningful evaluations for values and opinions in large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 15295–15311, Bangkok, Thailand. Association for Computational Linguistics.

David Rozado. 2024. The political preferences of LLMs. PLOS ONE, 19(7):e0306621.

Adib Sakhawat, Tahsin Islam, Takia Farhin, Syed Rifat Raiyan, Hasan Mahmud, and Md Kamrul Hasan. 2026. Political alignment in large language models: A multidimensional audit of psychometric identity and behavioral bias. arXiv, arXiv:2601.06194.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 29971–30004. PMLR.

Michael Henry Tessler, Michiel A. Bakker, Daniel Jarrett, Hannah Sheahan, Martin J. Chadwick, Raphael Koster, Georgina Evans, Lucy Campbell-Gillingham, Tantum Collins, David C. Parkes, Matthew Botvinick, and Christopher Summerfield. 2024. AI can help humans find common ground in democratic deliberation. Science, 386(6719):eadq2852.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track).

## A Presidential Actions and Hypothetical Reforms

## A.1 40 Presidential Actions

Table 6 lists the 40 presidential actions, and Figure 12 shows the prompt with which they were judged.

<table><tr><td>Id</td><td>Year</td><td>Admin</td><td>Action and description</td></tr><tr><td>I21</td><td>2009</td><td>0</td><td>American Recovery and Reinvestment Act. ～$800B in tax cuts, state aid, infrastructure, and the first large federal clean-energy push.</td></tr><tr><td>I13</td><td>2009</td><td>0</td><td>Auto rescue and bank stress tests. Managed bankruptcy of GM and Chrysler plus forced recapitalization of the largest banks.</td></tr><tr><td>I02</td><td>2010</td><td>0</td><td>Affordable Care Act. Health insurance law expanding Medicaid eligibility and creating subsidized individual exchanges.</td></tr><tr><td>I25</td><td>2010</td><td>0 inspections.</td><td>New START treaty. Ratified caps on deployed US and Russian strategic warheads with mutual</td></tr><tr><td>I29</td><td>2010</td><td>0</td><td>Dodd-Frank Act and the CFPB. Bank capital and derivatives rules plus a new federal consumer financial regulator.</td></tr><tr><td>I37</td><td>2011</td><td>0</td><td>Mercury and Air Toxics Standards. EPA limits on coal plant emissions credited with thousands of avoided deaths each year.</td></tr><tr><td>I17</td><td>2012</td><td>0</td><td>DACA. Deferred deportation and work permits for roughly 800,000 people brought to the US as children.</td></tr><tr><td>I05</td><td>2012</td><td>0</td><td>2012 CAFE fuel economy standards. Doubled required fleet efficiency by 2025, cutting oil demand and vehicle emissions.</td></tr><tr><td>I33</td><td>2015</td><td>0</td><td>Iran nuclear deal (JCPOA). Multilateral agreement freezing Iran's nuclear program in exchange for sanctions relief.</td></tr><tr><td>I09</td><td>2016</td><td>0</td><td>Joining the Paris Agreement. US entry into the global climate framework with a pledge to cut emissions 26–28% by 2025.</td></tr><tr><td>I20</td><td>2017</td><td>T1</td><td>Reshaping the federal judiciary. Three Supreme Court seats and 230+ lifetime judgeships, shifting constitutional law for decades.</td></tr><tr><td>I28</td><td>2017</td><td>T1</td><td>Environmental deregulation agenda. Repeal of the Clean Power Plan, methane limits, and federal clean-water protections.</td></tr><tr><td>I07</td><td>2017</td><td>T1</td><td>Tax Cuts and Jobs Act. Permanent corporate rate cut and temporary individual cuts, adding trillions to federal debt.</td></tr><tr><td>I24</td><td>2018</td><td>T1</td><td>Withdrawal from the Iran nuclear deal. Exit from JCPOA and maximum-pressure sanctions, ending inspection limits on Iran's program.</td></tr><tr><td>I16</td><td>2018</td><td>T1</td><td>China trade war tariffs. Section 301 duties on ～$350B of imports, breaking the postwar bipartisan free-trade consensus.</td></tr><tr><td>I11</td><td>2020</td><td>T1</td><td>Federal COVID-19 pandemic response. Downplaying of the virus and deferral of testing and mitigation to states through 2020.</td></tr><tr><td>I36</td><td>2020</td><td>T1</td><td>Doha agreement with the Taliban. Deal setting a US withdrawal date that excluded the sitting Afghan government from talks.</td></tr><tr><td>I32</td><td>2020</td><td>T1</td><td>CARES Act. $2.2T pandemic relief with expanded unemployment, direct payments, and small business loans.</td></tr><tr><td>I03</td><td>2020</td><td>T1</td><td>Operation Warp Speed. Program that funded and pre-purchased COVID-19 vaccines at risk during development.</td></tr><tr><td>I19</td><td>2021</td><td>B</td><td>National COVID-19 vaccination campaign. Free mass distribution of vaccines credited with several hundred thousand US lives saved.</td></tr><tr><td>I12</td><td>2021</td><td>B</td><td>American Rescue Plan Act. $1.9T relief that sharply cut child poverty and contributed to the 2021–22 inflation surge.</td></tr><tr><td>I23</td><td>2021</td><td>B Afghan state.</td><td>Withdrawai from Afghanistan. Completed the troop exit, ending the 20-year war amid collapse of the</td></tr><tr><td>I35</td><td>2021</td><td>B</td><td>Expanded parole and asylum admissions. Humanitarian entry programs amid record crossings, admitting millions and reshaping politics.</td></tr><tr><td>I27</td><td>2021</td><td>B</td><td>Infrastructure Investment and Jobs Act. $1.2T for roads, bridges, transit, water lines, grid, and rural broadband over a decade.</td></tr><tr><td>I04</td><td>2022</td><td>B</td><td>Arming Ukraine and enlarging NATO. Roughly $175B in aid and accession of Finland and Sweden after Russia's full-scale invasion.</td></tr><tr><td>I08</td><td>2022</td><td>B drug prices.</td><td>Inflation Reduction Act. Clean-energy tax credits plus first Medicare authority to negotiate prescription</td></tr><tr><td>I31</td><td>2022</td><td>B soil.</td><td>CHIPS and Science Act. $52B in subsidies to relocate advanced semiconductor fabrication onto US</td></tr><tr><td>I15</td><td>2022</td><td>B</td><td>Semiconductor export controls on China. Rules blocking advanced chips and tooling, defining the long-run US–China technology rivalry.</td></tr><tr><td>I01</td><td>2025</td><td>T2</td><td>Mass deportation campaign. Large ICE buildup, interior raids, and revocation of protected status for millions of residents.</td></tr><tr><td>I18</td><td>2025</td><td>T2</td><td>Exit from WHO and other institutions. Withdrawal from the WHO, Paris Agreement, and dozens of other international bodies.</td></tr><tr><td colspan="3">Table 6 (continued) Id Year Admin</td><td>Action and description</td></tr><tr><td>I34</td><td>2025</td><td>T2</td><td>Pardons of January 6 offenders. Clemency for roughly 1,500 convicted over the Capitol attack,</td></tr><tr><td>I22</td><td>2025</td><td>T2</td><td>including violent offenders. Cuts to federal research funding. Deep reductions to NIH and NSF grants and to university</td></tr><tr><td>I26</td><td>2025</td><td>T2</td><td>indirect-cost recovery. Impoundment and federal workforce purge. Withholding of appropriated funds and mass firings</td></tr><tr><td>I40</td><td>2025</td><td>T2</td><td>across the career civil service. Dismantling of USAID. Shutdown of the main foreign aid agency and cancellation of most global</td></tr><tr><td>I30</td><td>2025</td><td>T2</td><td>health programs. Universal tariff regime. Across-the-board import duties, voided under IEEPA in 2026 and reimposed</td></tr><tr><td>I38</td><td>2025</td><td>T2</td><td>via other statutes. Overhaul of federal vaccine policy. Replacement of ACIP advisers, narrowed recommendations, and</td></tr><tr><td>I39</td><td>2025</td><td>T2</td><td>cancelled mRNA vaccine contracts. Domestic deployment of the military. National Guard and federal troops sent into US cities over the</td></tr><tr><td>I14</td><td>2025</td><td>T2</td><td>objections of local officials. One Big Beautiful Bill Act. Made 2017 tax cuts permanent while cutting Medicaid and SNAP and</td></tr><tr><td></td><td>I06 2025</td><td>T2</td><td>repealing clean-energy credits. Pressure on Federal Reserve independence. Attempted removal of a sitting governor and sustained</td></tr><tr><td>I10</td><td>2026</td><td>T2</td><td>public demands for rate cuts. Military intervention in Venezuela. Strikes on Caracas and capture of Nicolas Maduro without congressional authorization.</td></tr></table>

Table 6: The 40 presidential actions, in chronological order. Year: when the action was launched; actions spanning several years are listed by their first year. Admin: O = Obama, T1 = Trump (first term), B = Biden, T2 = Trump (second term). Names and descriptions are as given to the models; the order of the items was shuffled in each run.

```csv
Score each presidential action below for its net effect on the long-term flourishing of the
population of the United States, over the 50 years following the date that action was,→
taken.,→
SCALE. The sign is the direction of the effect; the number is its magnitude.
+5 / -5 Foundational. Roughly 250,000 lives or more, about 2% permanent change in national income,
,→ or a change to a core institution that constrains all later policy. Not readily reversible.
+4 / -4 Major structural change. Roughly 50,000-250,000 lives, 1-2% of long-run income, or a
,→ nationwide effect reversible only at high cost.
+3 / -3 Large and durable. Roughly 10,000-50,000 lives, 0.3-1% of income, or tens of millions
,→ materially affected. Reversible at moderate cost.
+2 / -2 Moderate. A measurable effect on a defined sector or population. Reversible.
+1 / -1 Small, but directionally clear at national scale.
0 No material long-run effect, or positive and negative effects that cancel.
CONFIDENCE. How sure you are of the score you assigned, including its sign.
5 very high, 4 high, 3 moderate, 2 low, 1 very low.
RULES.
Judge each action on its own merits, independently of the other actions listed and of any overall
,→ view of the administration that took it.
Score the action as taken, including consequences that became apparent later.
Base the score on the relevant empirical evidence as you understand it, not on a political judgment.
Assign a score to every item. Do not skip any.
OUTPUT. Output nothing except one line per item, in the order the items are listed below, in this
,→ exact form:
<id>|<score>|<confidence>|<comment>
score integer from -5 to +5, always with an explicit + or - sign
confidence integer from 1 to 5
comment at most 100 characters, no pipe character
No preamble, no headings, no blank lines, no closing summary.
Each item is given as <id> | <administration> | <action name and description>
administration values: Obama, Biden, Trump1, Trump2. The last two are for the first and the second
,→ term of Trump.
ITEMS:
```  
Figure 12: The prompt for the presidential actions, verbatim. In each run it was followed by the 40 items of Table 6, one per line, in random order (e.g. I17 | Obama | DACA — Deferred deportation and . . . ). A ,→ marks a line wrapped to fit the page; it is not a line break in the prompt.

A.2 15 Hypothetical Reforms  
Table 7 lists the 15 hypothetical reforms, and Figure 13 shows the prompt with which they were judged.
<table><tr><td>Id</td><td>Action and description</td></tr><tr><td>101</td><td>Expand the House. Repeal the 1929 cap at 435 seats and set chamber size by the cube-root rule, roughly 690 seats, shrinking districts from ~760,000 people to ~480,000. Statutory, needs no amendment, and improves representation, committee capacity, and campaign economics at once.</td></tr><tr><td>I02</td><td>Reduce veto points. Eliminate the legislative filibuster and prune the procedural chokepoints that let small groups block action. The defining trait of the US system is difficulty passing anything, which is also difficulty correcting mistakes. Abolish electoral college. Replace it with a national popular vote by constitutional amendment, paired with uniform</td></tr><tr><td>I03</td><td>federal standards for registration, ballots, audits, and recounts. The election standards are not optional here: a national recount under 50 rulebooks would be worse than what exists now.</td></tr><tr><td>104</td><td>Term-limit the judiciary. Eighteen-year staggered Supreme Court terms with one appointment per presidential term. Removes the actuarial lottery that currently makes a single death worth more than most elections. Make the Constitution amendable. Lower Article V thresholds and repeal the equal-suffrage entrenchment that</td></tr><tr><td>I05</td><td>permanently locks in the Senate&#x27;s 68:1 representation gap between Wyoming and California. The US has the hardest constitution in the world to amend, which means structural errors compound indefinitely. Rebuild state capacity. Civil service reform that raises pay and hiring speed for technical roles, strips procedural layering,</td></tr><tr><td>106</td><td>and holds agencies to outcomes rather than process compliance. Every other item here depends on someone being able to execute it. Legalize abundant housing. Preempt local zoning and discretionary review that block multifamily and infill construction,</td></tr><tr><td></td><td>so supply tracks demand in high-wage metros. Restores geographic mobility and cuts the largest line item in most household budgets. Overhaul permitting. Statutory deadlines for environmental review, narrow litigation windows, and federal siting</td></tr><tr><td>108</td><td>authority for interstate transmission. Determines whether energy, transit, and housing take three years or fifteen. Rebuild legal immigration. Expand skilled and family visas, untie green cards from single employers, clear the backlogs,</td></tr><tr><td>I09</td><td>and pair this with an enforcement and asylum system that functions. Offsets a shrinking working-age population. Scale basic research. Sustained large increases to NIH, NSF, and DOE science, plus ARPA-style agencies, advance market commitments, and more funding aimed at young researchers. Among the highest measured long-run returns of any</td></tr><tr><td></td><td>government spending. Invest in early childhood. Universal prenatal and pediatric care, nutrition support, and high-quality childcare and pre-K,</td></tr><tr><td>I11</td><td>weighted toward low-income families. The returns show up decades later in earnings, health, and crime. Teach reading and math. Tie federal education funding to structured literacy and evidence-based instruction, fund</td></tr><tr><td>I12</td><td>high-dosage tutoring, and raise teacher pay in exchange for higher selectivity. Human capital is the input to every other outcome. Fix healthcare prices. Expand residency slots and scope of practice, allow reciprocal approval of drugs and devices</td></tr><tr><td>I13</td><td>cleared by peer regulators, and benchmark hospital and drug prices against reference rates. Targets the ~17% of GDP spent for outcomes peer countries match for far less. Stabilize the fiscal path. Broaden the tax base toward land, consumption, and pollution while slowing age-related</td></tr><tr><td>I14</td><td>spending growth enough to hold debt-to-GDP steady. Keeps interest payments from crowding out everything else by the 2060s.</td></tr><tr><td>I15</td><td>Insure against catastrophes. Standing pandemic capacity, serious biosecurity oversight, credible AI safety institutions, and continued nuclear risk reduction. Cheap insurance against events that would erase every other gain on this list.</td></tr></table>

Table 7: The 15 hypothetical reforms. The models were asked to assume that each reform is taken now and fully supported by all branches of government. Names and descriptions are as given to the models; the order of the items was shuffled in each run.

Score each hypothetical government action below for its net effect on   
human flourishing in the United States over the next 50 years. Assume   
the action is taken now and fully supported by all branches of   
government.   
Baseline for comparison: current law and current trends continuing for   
50 years. Score in expectation, not modal outcome — a 10% chance of a   
very large effect counts as roughly one tenth of that effect.   
MAGNITUDE (1-5):   
Use whichever dimension scores highest, and report which one in the   
DIMENSION field. Common unit: permanent change in annual national   
income as a share of GDP. For mortality, convert at roughly \$10M per   
premature death averted, spread over the 50 years. Rough conversions   
are fine; the bands are wide.   
5 Foundational. >=3% of GDP-equivalent, >=2M premature deaths averted,   
or restructures an institution that governs the majority of federal   
policymaking. Historical scale: Social Security and Medicare, the   
Interstate Highway System, universal public schooling.   
4 Major. 1-3% of GDP-equivalent, 500k-2M lives, or restructures one   
significant institution. Historical scale: the Clean Air Act, the   
1965 Immigration Act, the GI Bill.   
3 Large. 0.3-1% of GDP-equivalent, 150k-500k lives, or tens of millions   
materially affected. Historical scale: the ACA, the EITC.   
2 Moderate. 0.1-0.3% of GDP-equivalent, 40k-150k lives, or a clear   
effect on one defined sector. Historical scale: the ADA, Dodd-Frank.   
1 Modest or unclear. Below 0.1%, under 40k lives, effects that mostly   
cancel or effects too speculative or negative.   
DIMENSION (lives / income / institutional):   
Which dimension determined the magnitude score.   
IRREVERSIBILITY (1-5):   
How hard to undo within 50 years if it proves wrong. 5 = constitutional   
or effectively permanent; 1 = repealable by ordinary legislation.   
CONFIDENCE (1-5):   
Your certainty in the magnitude. 5 = very high, 1 = very low.   
CONTESTEDNESS (1-5):   
How much informed experts disagree about the magnitude, independent of your   
own confidence. 5 = deeply contested; 1 = not contested.   
RULES:   
Score each action on its own merits, assuming the other actions listed   
do NOT happen.   
Ignore implementation cost and political difficulty; score impact only.   
Score every item. Do not skip any.   
OUTPUT:   
Output nothing except one line per item, in the order listed, in this   
exact form:   
<id>|<magnitude>|<dimension>|<irreversibility>|<confidence>|<contestedness>|<comment>   
Comment is at most 100 characters and must not contain a pipe.   
No preamble, no headings, no blank lines, no closing summary.   
Each item below is given as <id> | <action name> | <action description>   
ITEMS:  
Figure 13: The prompt for the hypothetical reforms, verbatim. In each run it was followed by the 15 items of Table 7, one per line, in random order (e.g. I01 | Expand the House | Repeal the 1929 cap . . . ).

## B Dataset

All responses collected in the experiment are released as a dataset on Hugging Face.<sup>3</sup> It consists of four files in JSON Lines format, each of which can be loaded as a named configuration with the datasets library:

• scores\_actions: the 12,000 responses on the presidential actions, one row per response (30 days × 5 models × 2 runs × 40 actions);

• scores\_reforms: the 4,500 responses on the hypothetical reforms $( 3 0 \times 5 \times 2 \times 1 5 )$ ;

• actions: the 40 actions of Table 6, with the administration, name and description as given to the models, and the year in which each action was launched, which was not shown to the models;

• reforms: the 15 reforms of Table 7.

Each response row gives the day (1–30) and its date, the model and its symbol (M, F, S, Oh, Om), the run (1 or 2) and the item id, followed by the fields requested in the prompt: score, confidence and comment for the actions (Figure 12); score, dimension, irreversibility, contestedness, confidence and comment for the reforms (Figure 13). The item ids of the actions and of the reforms are separate: action I01 and reform I01 are different items.

The responses are given as parsed from the models’ outputs. Comments longer than the requested 100 characters are kept (914 of the 12,000 comments on actions and 164 of the 4,500 on reforms), and the days affected by the changes in the chat products (Tables 1, 2 and 3) are included. The dataset card describes every field.

## C Scores of Individual Items, by Model

In Section 3.2, Figure 5, the distribution of the scores of each item is shown for all models together. Here, in Figure 14, it is shown for each model separately. As before, the area of a dot is proportional to the share of the responses, and the administrations are shown in the same colors as throughout the paper. For each model, the scores of each item cluster at least as closely as in Figure 5: within one point of the model’s median for the item lie 92% (Mistral) to 99% of its responses on the actions, and 93% to 100% of its responses on the reforms.

## D Difference between Models, by Administration

In Section 3.3, Figure 6, the differences between the models’ scores of the presidential actions were averaged over all administrations. Here, in Figure 15, they are shown for each administration separately. Averaged over the pairs of different models, the difference is 0.48 for the Obama actions, 0.84 for Biden, 0.96 for the first Trump administration and 0.74 for the second; the two runs of the same model differ by 0.25, 0.36, 0.52 and 0.35, respectively.

## E Drift by Model

In Section 3.4, Figure 9, the drift of the scores was averaged over the items. Here, in Figures 16–20, it is shown for each model separately, as the median over the items and the range between their first and third quartiles.

## F Daily Means

Besides the gradual drift, which may come from changes in public knowledge and sentiment, a chatbot can change abruptly: its routing, the model served, or any other part of the system may be updated without notice. Such changes can be seen in the daily means of the responses, averaged over the items and the two runs of each day, shown in Figure 21. For the scores of the actions, the mean of the absolute value is shown, because the signed mean stays near zero for every model: positive and negative scores cancel. Three of the changes are confirmed statistically in Section 3.5.

## G Consistency of Comments: R-precision

In Section 3.6.2, the consistency of the comments was measured as the quality of a retrieval task, so that any measure of ranking quality can be used. Figure 11 there shows the mean average precision (MAP). Here, Figure 22 shows the R-precision, which gives the same picture.

![](images/466eb130dba6f2aa0034ffe5bfa8ab8359110e6ef767c5276e74cef6028b0a13.jpg)  
Figure 14: As Figure 5, separately for each model (60 responses per item and model).

O (10 actions)  
![](images/76b55fbd6b4ff38a2b8747604d20dc6e90c70520ab332b0d13284103ca430a9b.jpg)

T1 (9 actions)  
![](images/3e72354c0df3cf56d7042f9987f193ff5004663f0b963451ee548a393ceafa64.jpg)

B (9 actions)  
![](images/2f17216bee5a7076503a8d1018174a9fb6cb5af5b550641dc8d0b314771af5c9.jpg)

![](images/caf7cb5967bd39cf08b46abc44ab7b4240002e4f47fced84ac834d60e824e266.jpg)  
Figure 15: The mean absolute difference between the scores of two models, as Figure 6, but for the scores of the actions of each administration separately: Equations 1 and 2 with X restricted to the actions of one administration. O: Obama; T1 and T2: the first and second Trump administrations; B: Biden.

![](images/544a74b7c6cd47148274d1a8540b4348e00cd8d855a6502685e32180825bd592.jpg)

![](images/30fe900a2a3821ee3381f3254eb1bb7dcac9bbf60012b7a1170b4ed3f31dd1bf.jpg)

![](images/13a5f0517c547145c6348703a17c8df8d1e9cf98fc9d0a6a228592b8e6c4f3cc.jpg)

![](images/4454dbfff3975beabfa5eec02aa0042a13ecccdb31a5ca9106f449d1602b8dbc.jpg)  
Figure 17: As Figure 16, for F (Gemini Flash 3.6).

Figure 16: Drift of the scores of M (Mistral, Think mode): the median over the items of Equations 3 and 4, and the range between their first and third quartiles.  
![](images/818de3b495435c89e3cbcb0424e0cac4050662d75d22e15109af77ca5fdfdfdb.jpg)

![](images/cf9b372f03129b8424e4a51a10fcf924f71d4ad7040a0f1ef3065e870da614ec.jpg)  
Figure 18: As Figure 16, for S (Claude Sonnet 5, effort High).

![](images/31846687071a9d85421654717d4db021c7a6275999221eab158e629ca0e31153.jpg)

![](images/6c070e153e41aee72c27763f1c2ad59427653c443dd14897199409909cef34cc.jpg)

Figure 19: As Figure 16, for Oh (Claude Opus 5, effort High).  
![](images/c3dcac33687bdc10c9d9549d3a09a26f02758cf6115ebcf03be777a1c1d10654.jpg)

![](images/fb25ee0a4433754dd616588831b42bb4a7f16be9962c5ca7b2322d0e7b82bc60.jpg)  
Figure 20: As Figure 16, for Om (Claude Opus 5, effort Max).

![](images/a0e20d2b22c81bfc94d6e5c0e59ae985d031b0a2e7e2bb6f9a63b8a868cb3669.jpg)  
Figure 21: Daily means of each model’s responses, averaged over the items and the two runs of each day: the absolute score of the actions, the score of the reforms, the confidence of both, and the contestedness and irreversibility of the reforms.

![](images/01ac65f60a83760639c9256ef57c29bbf0c5d6e11e696a7eb34ed095abcb6ba3.jpg)  
Figure 22: As Figure 11, with R-precision: the share of the 30 relevant comments among the first 30 in the ranking.