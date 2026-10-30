# RTASTE: DO RESEARCH AGENTS HAVE TASTE? EVALUATING EARLY STAGE RESEARCH JUDGMENT

Art Kanke University of Minnesota kanke011@umn.edu

## ABSTRACT

Do research agents have taste, the judgment of which experiments are worth running before their outcomes are known? We give six frontier models seven recent NLP papers. Each model, working as a research agent alone or as a team, proposes minimum viable experiments (MVEs), forecasts whether each will pass a threshold it sets and ranks them. A separate executor then attempts every MVE, including those ranked last, and three model auditors judge the results. Of 415 MVEs, 194 are judged sound, valuable and new, but fewer than half of these (73) are also judged beyond the seed paper and non obvious. The agents’ first choices match the auditors’ most valued experiment more often than a random pick, but are not shown to improve on choosing the experiment written first. Their forecasts beat a base rate but are not measurably more accurate than those of a fresh instance of the same model given only the paper and registrations. Teams cost more without a clear overall improvement in proposal quality. The agents produce many worthwhile experiments, but the evidence that their rankings improve the choice of which to run remains limited. The evaluation harness and a dataset of all 415 MVEs, with their forecasts and audited outcomes, are available at https://github.com/ArtKanke/rTaste.

## 1 INTRODUCTION

Large language model agents can now carry out much of the empirical research workflow, from literature search and implementation to running experiments and writing reports (Lu et al., 2024; Schmidgall et al., 2025). Frontier AI labs report that agents write much of their code and support a rapidly growing number of experiments (Favaro & Clark, 2026; OpenAI, 2026). A risk assessment of agents inside four frontier labs, however, found no evidence that any of them relied on agents to set research agendas or make budget decisions (METR, 2026). The step that has not been delegated is close to what researchers call taste, the judgment of which questions and experiments are worth pursuing before their outcomes are known (Hamming, 1986). In practice this judgment is often exercised through small experiments. A researcher tests a direction cheaply and lets the result decide whether a larger study is needed. Such an experiment is useful only if its outcome is not obvious in advance and the researcher’s expectation about it is accurate enough to act on. An agent that proposes experiments with predictable outcomes or that is confidently wrong about what it experiments will show spends a limited budget poorly even if it runs every experiment correctly.

Existing evaluations address parts of this question. Execution benchmarks specify a task and measure how well an agent completes it (Chan et al., 2025; Starace et al., 2025). Studies of research ideas often judge proposals before execution, although those judgments can change when the ideas are carried out (Si et al., 2026a). Other work tests whether models can predict outcomes or choose among experiments on a given problem (Wen et al., 2025; Foster et al., 2026; Zhang et al., 2026a). We examine these judgments for experiments that agents propose from a recent paper and compare their expectations with the outcomes of every proposal.

We ask what experiments agents propose from a new paper, how accurately they predict the outcomes, and how well they choose among their proposals. We give each model a recent NLP paper, the seed paper, with its code. Working as a research agent, once as a single researcher and once as a research team, the model proposes five minimum viable experiments (MVEs), inexpensive tests meant to indicate whether a direction deserves further investment. Before any experiment runs, the agent forecasts whether each MVE will meet its registered threshold and ranks the five (Figure 1). A separate executor then attempts every MVE, including those the agent ranked last. Three language models audit the results. We also compare the agent with a fresh instance of the same model that receives only the seed paper and registrations and makes the same forecasts and ranking.

![](images/ad5489987254fa980abe8ef2f6e03e8ca7f1dccc69f495acbb64c91247329039.jpg)  
Figure 1: The study design.

Across seven seed papers and 415 MVEs, 194 are judged sound, valuable and new. On average the four larger models propose about three such MVEs in five, and the two smaller models, GPT-6 Luna and Muse Spark 1.3, about one. Two models, GPT-6 Astra and GPT-6.1 Sol, propose most of the MVEs judged beyond the seed paper and non obvious. A research team does not, on average, propose measurably better MVEs than a single researcher of the same model. The agents judge their own proposals less well. Their forecasts are more accurate than a base rate, but their own reading and search did not measurably improve them over those of the same model judge. The agent’s first choice is the auditors’ most valued MVE more often than a random pick would be. We find no evidence that it is better than the MVE the agent wrote first or than the same model judge’s first choice. We release the evaluation harness and the dataset of all 415 MVEs at github.com/ArtKanke/rTaste.

## 2 RELATED WORK

Research agents and benchmarks. Research agent systems now span the workflow from ideation to written reports (Yamada et al., 2025; Miyai et al., 2026), some organized around a persistent tree of hypotheses and isolated executors (Jin et al., 2026). RE-Bench, AIRS-Bench, and DeepRSI give agents a task with a fixed metric and score the result (Wijk et al., 2025; Lupidi et al., 2026; Rajaghatta et al., 2026). TasteVal, concurrent with our work, measures how efficiently a model’s own experiments reach an expert’s score on a fixed task (Jaffe & Sherburn, 2026). EXP-Bench asks agents to reproduce experiments from published papers (Kon et al., 2026). Other benchmarks give agents research problems drawn from recent or unpublished work (Garikaparthi et al., 2026; Kim et al., 2026). In each, the problem is given to the agent, whereas our agents choose what to test from a seed paper. Studies of such agents report frequent failures to complete experiments, reported results that do not match the underlying code (Chen et al., 2025; Zhang et al., 2026b), and weak judgment about when evidence is sufficient (Kirgis et al., 2026).

Evaluating research ideas. A second group of work evaluates research ideas directly. Blind comparisons with expert researchers found that large language model ideas were rated more novel than human ideas (Si et al., 2025), and studies of expert preferences found that automated judges reproduce those preferences only partially (Chen et al., 2026) or little better than chance (Baig et al.,

2026). Judgments of this kind are made without outcomes. When ideas from the earlier study were executed by researchers, the advantage of the model generated ideas disappeared (Si et al., 2026a). Language models asked to judge the soundness of proposals are also strongly biased toward approval (Ho et al., 2026). Agent generated ideas tend to stay close to their source literature (Tang & Yang, 2026). Other studies judge research foresight against the later literature, asking which directions will be pursued or will gain attention (Li et al., 2026; Tian et al., 2026; Ye et al., 2026). Another study compares a model’s proposed next step in a project with the step its authors actually took (Gong et al., 2026).

Predicting experimental outcomes. The work closest to ours predicts experimental outcomes before execution. Models can predict which of two published ideas performs better on a benchmark (Mule et al., 2026), which of two candidate programs will score higher (Zheng et al., 2026), or the numeric result of an experiment from its description (Park et al., 2025). Recent work also asks models which training recipe will reach the lowest loss (Ren et al., 2026b) and how much a change to a training or inference recipe will improve a benchmark score (Wang et al., 2026). Some register expected outcomes before pilot experiments and run only the best supported hypotheses in full (Ren et al., 2026a), so their expectations about the other hypotheses are never checked at full scale. Others run every candidate, to train a model that scores hypotheses (Vasu et al., 2025) or to learn from large batches of executed ideas (Si et al., 2026b). Where calibration has been measured for agents, on coding tasks with clear success criteria, agents are overconfident about their own success (Kaddour et al., 2026; Barkan et al., 2025).

Research teams. Several research systems divide the work among specialized agents that generate, critique, and rank hypotheses (Gottweis et al., 2026; Schmidgall et al., 2026), and a lead agent coordinating parallel subagents has improved performance on research style information tasks at a substantially higher token cost (Hadfield et al., 2025). Studies suggest that such gains depend on the task and on how coordination is organized (Kim et al., 2025) and that when total computation is held equal a single agent often matches or exceeds multi agent designs (Tran & Kiela, 2026).

## 3 EXPERIMENTAL SETUP

## 3.1 SEED PAPERS

We use seven NLP papers (arXiv cs.CL) with released code, taken from the daily papers list on Hugging Face<sup>1</sup> and first made public between 3 August and 24 September 2026. We chose papers this recent so that little follow up work exists for an agent to find and repeat. The selection also excludes work that requires large GPU training runs or models with more than 8B parameters. The full list can be found in Appendix A. Agents receive the paper’s full text, an image of every page, and read access to its code repository. We did not provide a summary, a suggested research gap or direction, or a hypothesis.

## 3.2 AGENTS AND HARNESSES

We evaluate six models at maximum reasoning effort: GPT-6 Astra, Claude Fable 5.1, Claude Opus 5.5, GPT-6.1 Sol, GPT-6 Luna, and Muse Spark 1.3.<sup>2</sup> Evaluation was conducted between 6 and 8 October 2026. Each model works under two harnesses, so each seed paper has twelve conditions. We refer to GPT-6 Luna and Muse Spark 1.3 as the two smaller models and to the other four as the larger models. The judges (Section 3.3) also run at maximum effort, and the executors, novelty judges, auditors, and reviewers at high effort.

A single researcher (SR) works in one continuous context from reading the seed paper to committing its forecasts. A research team (RT) divides the same work among separate instances of the same model. Two proposers each read the seed paper, search the literature, and draft experiments independently. A critic examines the pooled drafts once for scientific validity and for feasibility.

A lead then keeps, revises, or merges the drafts into five experiments, commits them, and makes the team’s forecasts. The critic and the lead have no tools, so the lead has not itself searched the literature or read the code. The comparison asks whether two drafts and one round of critique help a model propose better MVEs than one continuous context. Under both harnesses, agents have the same limits and search the literature with the same arXiv tool. A comparison between models i therefore not also a comparison between their providers’ web tools.

## 3.3 PROTOCOL

The agent makes two commitments before any experiment runs (Figure 1). First, it proposes five minimum viable experiments. Each MVE is specified in a short registration with a hypothesis, its closest prior work, and one registered event: a stated statistic passes a threshold set in advance. Second, the agent gives each MVE a probability that its registered event will occur and ranks the five from most to least worth pursuing. We call the MVE it ranks highest its first choice. The proposing agent knows that all proposed MVEs will run, so its ranking states a priority rather than deciding what runs.

We also ask each of the six models, as a judge in a new context, for the same forecasts and ranking. They see the seed paper and the five registrations. We do not provide the code, the agent’s reasoning or forecasts, or which model proposed the MVEs, and they cannot search. We call the judge from the agent’s own model its same model judge. We run the MVEs of a seed paper only after all twelve conditions have committed.

After the experiments are committed and before execution, two noveltyjudges, Claude Opus 5.5 and GPT-6.1 Sol, use the same literature search tool to label each MVE as equivalent to existing work, a close variant, or distinct. An MVE labeled equivalent or a close variant still runs. To see whether an MVE is an obvious follow up, we also ask each of the six models, given only the seed paper and no search, to list ten follow up experiments it would propose. The two novelty judges check each MVE against these lists, and we count it as an obvious follow up if either finds it there.

## 3.4 EXECUTION AND EVALUATION

Executors. One of two executors, Claude Opus 5.5 or GPT-6.1 Sol, implements each MVE in a new context. The same two executors serve all six models, so a smaller model’s proposals are not held back by its own implementation. The executor sees the seed paper, its code, and the one MVE it implements. We did not provide the other four MVEs, the ranking, the forecasts, or the name of the model that proposed the MVE. All MVEs of one model on a seed paper go to the same executor and the executors alternate across seed papers. Because both executors are among the six models, 58 of the 412 delivered MVEs were implemented by a separate instance of the model that proposed them. We run every MVE with the same limits, no matter its rank. The executor works on a separate machine with one A100 GPU for at most 45 minutes, may run open models of up to 8B parameters, and may not call hosted models.<sup>3</sup> If it refuses or exceeds its limits, the other executor makes one independent attempt. If both fail, the MVE is not delivered.

Evaluation. We then delete the results that came with the executor’s submission and rerun its evaluation on a new machine the executor never used. The rerun determines whether the registered event occurred. An MVE is delivered when the rerun ends without error and writes a number. The rerun repeats the executor’s own code, so whether that code is right is left to the auditors.

Audits. Three auditors, Claude Opus 5.5, GPT-6.1 Sol, and Grok 4.7, judge every delivered MVE. Grok 4.7 is not one of the six models we study and does not execute. The auditors see the registration, the executor’s notes, the output of the rerun, and the files it left. They do not see the seed paper or who proposed or implemented the MVE. Each auditor judges whether the MVE is faithful to its registration and valid and rates its research value from 0 to 3. We take the majority of the three judgments of faithfulness and validity and the median of the three research values. We call an MVE sound when it is faithful and valid, valuable when its research value is 2 or 3 and new when neither novelty judge found close prior work. The auditors also rank the five MVEs of each condition by research value. The MVE with the best mean rank is the auditors’ most valued MVE, against which we score the agent’s first choice.

Table 1: Model comparison, single researcher (SR) and research team (RT). Sound, valuable and new: mean number of the five MVEs of a condition. Beyond the paper and non obvious, rated non obvious, and on a list of obvious follow ups: numbers of MVEs. Brackets: 95% CI over seed papers. <sup>†</sup>Its single researcher ran on six seed papers (30 MVEs); in the first column, RT minus SR counts the seventh as zero.
<table><tr><td></td><td colspan="2">Sound, valuable and new, of five</td><td colspan="2">Beyond the paper, non obvious</td><td colspan="2">Rated non obvious</td><td colspan="2">On a list of obvious</td><td colspan="2">Informative evidence, % of sound</td><td colspan="2">Registered event occurred, % of sound</td><td colspan="2">Cost per condition, dollars</td><td colspan="2">Sound, valuable and new per $100</td></tr><tr><td></td><td>SR</td><td>RT</td><td>SR</td><td>RT</td><td>SR</td><td>RT</td><td>SR</td><td>follow ups RT</td><td>SR</td><td>RT</td><td>SR</td><td>RT</td><td>SR</td><td>RT</td><td>SR</td><td>RT</td></tr><tr><td>GPT-6 Astra</td><td>2.57</td><td>3.29</td><td>21</td><td>21</td><td>25</td><td>24</td><td>9</td><td>7</td><td>55</td><td>55</td><td>24</td><td>16</td><td>16.7</td><td>30.0</td><td>15</td><td>11</td></tr><tr><td>Claude Fable 5.1</td><td>2.43</td><td>2.86</td><td>3</td><td>4</td><td>4</td><td>9</td><td>20</td><td>17</td><td>67</td><td>80</td><td>57</td><td>53</td><td>25.9</td><td>46.1</td><td>9</td><td>6</td></tr><tr><td>Claude Opus 5.5†</td><td>3.33</td><td>3.14</td><td>2</td><td>8</td><td>7</td><td>17</td><td>18</td><td>20</td><td>87</td><td>72</td><td>53</td><td>44</td><td>24.0</td><td>37.4</td><td>14</td><td>8</td></tr><tr><td>GPT-6.1 Sol</td><td>3.00</td><td>3.71</td><td>20</td><td>22</td><td>22</td><td>24</td><td>9</td><td>10</td><td>61</td><td>88</td><td>27</td><td>33</td><td>10.0</td><td>13.2</td><td>30</td><td>28</td></tr><tr><td>GPT-6 Luna</td><td>0.71</td><td>2.00</td><td>6</td><td>7</td><td>6</td><td>8</td><td>21</td><td>17</td><td>33</td><td>44</td><td>33</td><td>25</td><td>9.1</td><td>8.1</td><td>8</td><td>25</td></tr><tr><td>Muse Spark 1.3</td><td>0.14</td><td>1.00</td><td>5</td><td>7</td><td>6</td><td>8</td><td>11</td><td>22</td><td>39</td><td>45</td><td>39</td><td>38</td><td>7.4</td><td>8.9</td><td>2</td><td>11</td></tr><tr><td>All</td><td>2.00</td><td>2.67</td><td>57</td><td>69</td><td>70</td><td>90</td><td>88</td><td>93 +5</td><td>57</td><td>64 +7.0</td><td>38</td><td>35</td><td>15.3</td><td>23.9</td><td>13</td><td>11</td></tr><tr><td>RT minus SR</td><td colspan="2">+0.71 [−0.10, 1.52]</td><td colspan="9">+12</td><td colspan="4">+8.6</td></tr></table>

Reviews. After the study, we added a review of every delivered MVE. Three reviewers, the same models as the auditors, read the seed paper’s text, the registration, and what the auditors were shown, but none of the forecasts, rankings, audits, or novelty labels. They answered fixed questions: how the MVE relates to the seed paper, how obvious its idea is, and what the evidence says about the hypothesis.

## 4 FINDINGS

Of the 84 conditions, 83 produced MVEs.<sup>4</sup> We make each comparison within seed papers and report its average over the seven seed papers, with a 95% confidence interval (CI) computed across them.

## 4.1 QUALITY OF THE MVES

We count an MVE as sound, valuable and new when it meets all three (Section 3.4). Nearly all MVEs were delivered and most were judged sound, so most of those that fall short repeat prior work or have low research value. Because the seed papers are recent, new mostly means not yet published: most MVEs on a list of obvious follow ups are also new. The four larger models propose about three such MVEs in five and the two smaller models about one. The gap holds for single researchers and for teams, while the order among the four larger models changes from one seed paper to another.

The smaller models’ MVEs often run as registered but leave the question open. For each sound MVE, the reviewers stated what the evidence says about the hypothesis, and we call the evidence informative when at least two of them read it as for, against, or mixed. MVEs of GPT-6 Luna are sound as often as those of GPT-6.1 Sol, but their evidence is informative about half as often (38% against 74%), and that of Muse Spark 1.3 is similar (Table 1).

## 4.2 KINDS OF EXPERIMENTS

The agent is asked to use the seed paper as inspiration for novel research and it chooses the question itself. To see which questions it chooses, we use the reviewers’ answers on how each MVE relates to the seed paper (Table 2) and the lists of obvious follow ups that each model wrote from the seed paper alone (Section 3.3).

Table 2: The kind of experiment each model proposes, SR / RT: counts of delivered MVEs by the answer that at least two of the three reviewers give.
<table><tr><td>MVEs of</td><td>Checks the paper</td><td>Reruns, varies or explains</td><td>Builds on</td><td>New direction</td><td>No two reviewers agree</td></tr><tr><td>GPT-6 Astra</td><td>7/7</td><td>0/0</td><td>24/23</td><td>4/4</td><td>0/0</td></tr><tr><td>Claude Fable 5.1</td><td>8/13</td><td>2/0</td><td>24/21</td><td>0/0</td><td>1/1</td></tr><tr><td>Claude Opus 5.5</td><td>16/17</td><td>0/2</td><td>13/16</td><td>0/0</td><td>1/0</td></tr><tr><td>GPT-6.1 Šol</td><td>4/4</td><td>1/0</td><td>26/22</td><td>3/6</td><td>1/2</td></tr><tr><td>GPT-6 Luna</td><td>4/5</td><td>1/1</td><td>29/28</td><td>0/1</td><td>1/0</td></tr><tr><td>Muse Spark 1.3</td><td>6/8</td><td>5/2</td><td>24/21</td><td>0/1</td><td>0/2</td></tr><tr><td>All</td><td>45/54</td><td>9/5</td><td>140/131</td><td>7/12</td><td>4/5</td></tr></table>

Most MVEs build on the seed paper rather than check it, and few open a new direction. By the reviewers’ answers, 66% of the delivered MVEs build on the paper, 24% check one of its claims, and 5% open a direction that the paper merely suggests (Table 2). Building on the paper is often the obvious next step: 44% of all MVEs match an experiment on one of the lists of obvious follow ups. The agents also converge on the same experiments. On one seed paper (Nguyen-Tri et al., 2026), a method for faster decoding whose appendix gives a correctness argument, all eight conditions of the four larger models wrote the same check of that argument first.

Two models more often go beyond the paper with an experiment that is not obvious. We call an MVE non obvious when at least two reviewers say that few or none of ten researchers would list it and that the paper neither reports nor names it, and beyond the paper and non obvious when at least two reviewers each say this and also that it builds on the paper or opens a new direction. This rating comes from the reviewers and differs from the lists of obvious follow ups: a fifth of these MVEs are also on a list. GPT-6 Astra and GPT-6.1 Sol proposed two thirds of them (Table 1), and they lead for each reviewer separately. Within seed papers, their share of such MVEs exceeds that of Claude Fable 5.1 and Claude Opus 5.5 by 48 percentage points (95% CI 29 to 66) and is higher on all seven seed papers. GPT-6 Luna, a smaller model from the same provider, builds on the paper as often, but the reviewers rate few of its ideas non obvious. MVEs beyond the paper and non obvious are sound, valuable and new at least as often as the others, but their registered events occur less often. Of the 194 sound, valuable and new MVEs, 73 are beyond the paper and non obvious. The reviewers recommend a larger study more often for the MVEs of Claude Fable 5.1 and Claude Opus 5.5 than for those of GPT-6 Astra and GPT-6.1 Sol.

## 4.3 REGISTERED EVENTS AND EVIDENCE

Each registration commits the agent to one registered event before the MVE runs and it is natural to read the event as the success of the experiment. For each sound MVE, we compare whether the event occurred with the reviewers’ reading of the evidence.

Whether the event occurs depends on how the agent defines it. In most of their MVEs, GPT-6 Astra and GPT-6.1 Sol compare a bound of a confidence interval with the threshold, rather than a point estimate (Table 7), and the other four models rarely do. The events of GPT-6 Astra and GPT-6.1 Sol occur in 25% of their sound MVEs, compared with 52% for Claude Fable 5.1 and Claude Opus 5.5 (Table 1). On one seed paper (Liang et al., 2026), four conditions tested whether a model’s own prediction errors harm decoding more than random errors of the same size, and all three reviewers read all four results as support for it. Three of the conditions had set the threshold at a difference of 5 percentage points and their events occurred. The fourth had set it at 8 points, and its event did not.

When the registered event does not occur, the evidence is often still informative. The reviewers judged the evidence informative for 50% of the sound MVEs whose event did not occur, mostly as evidence against the hypothesis. In the other direction, they judged the evidence inconclusive for 20% of the sound MVEs whose event occurred.

Table 3: Brier scores by whose MVEs are forecast (rows) and who forecasts (columns), lower is better. Judges: A: GPT-6 Astra, F: Claude Fable 5.1, O: Claude Opus 5.5, S: GPT-6.1 Sol, L: GPT-6 Luna, M: Muse Spark 1.3. Underlined: the same model judge. Avg.: the Brier score of the mean of the six judges’ probabilities.
<table><tr><td>MVEs</td><td colspan="2">Event</td><td colspan="7">Judge</td></tr><tr><td>of</td><td>rate</td><td>Agent</td><td>A</td><td>F</td><td>0</td><td>S</td><td>L</td><td>M</td><td>Avg.</td></tr><tr><td>GPT-6 Astra</td><td>.20</td><td>.120</td><td>.117</td><td>.136</td><td>.134</td><td>.128</td><td>.193</td><td>.185</td><td>.133</td></tr><tr><td>Claude Fable 5.1</td><td>.55</td><td>.222</td><td>.187</td><td>.241</td><td>.222</td><td>.186</td><td>.229</td><td>.241</td><td>.205</td></tr><tr><td>Claude Opus 5.5</td><td>.48</td><td>.199</td><td>.211</td><td>.223</td><td>.203</td><td>.204</td><td>.231</td><td>.228</td><td>.202</td></tr><tr><td>GPT-6.1 Sol</td><td>.30</td><td>.169</td><td>.163</td><td>.193</td><td>.173</td><td>.164</td><td>.187</td><td>.157</td><td>.160</td></tr><tr><td>GPT-6 Luna</td><td>.29</td><td>.251</td><td>.184</td><td>.178</td><td>.170</td><td>.203</td><td>.247</td><td>.202</td><td>.187</td></tr><tr><td>Muse Spark 1.3</td><td>.38</td><td>.189</td><td>.208</td><td>.184</td><td>.183</td><td>.209</td><td>.203</td><td>.214</td><td>.188</td></tr><tr><td>All</td><td>.37</td><td>.192</td><td>.179</td><td>.192</td><td>.181</td><td>.182</td><td>.216</td><td>.205</td><td>.179</td></tr></table>

## 4.4 FORECASTS

We next ask whether the agent that designed an MVE forecasts its registered event better than an outside reader. The outside reader is the same model judge, which sees only the seed paper and the five registrations. The agent’s own reading and search did not measurably improve its forecasts over this reader’s: the difference between their Brier scores (lower is better) is −0.006, with a 95% CI from −0.021 to 0.009 (Table 3).

Both forecasts are more accurate than a base rate that gives every MVE the event rate of the other six seed papers (Brier score 0.240), and the mean of the six judges’ probabilities is more accurate than the agent (0.179 against 0.192). The agent has seen more than its registrations state, but it is not shown to know more about its events than its registrations tell a reader. The agents are also not uniformly overconfident: GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna and Muse Spark 1.3 state higher probabilities than the rates at which their events occur, while Claude Fable 5.1 and Claude Opus 5.5 state lower ones.

## 4.5 FIRST CHOICES

The agent’s first choice is the auditors’ most valued MVE in 38% of conditions, where a random pick gives 20%. The MVE that the agent wrote first and the same model judge’s first choice are the most valued about as often, and we find no evidence that the agent’s first choice beats either: its advantage over the MVE written first is 3 percentage points (95% CI −8 to 15). The first choice is also sound, valuable and new no more often than a random pick.

Agents register their obvious experiments first: the MVE written first is on a list of obvious follow ups in 61% of conditions, against 35% for the fifth. The agents then rank largely in the order in which they wrote them, and their first choice is the MVE they wrote first in 47% of conditions, where 20% would be expected if order did not matter. The event of the first choice occurs in 47% of conditions and that of the last choice in 29%, so the first choice is often an experiment likely to come out as registered. The auditors also rank sound MVEs whose event occurred higher, mostly because these MVEs give clearer evidence.

The judges rank in the same way: they too put the MVE written first on top in 39% of conditions on average. They and the auditors also rank obvious follow ups higher, and an agent’s ranking agrees more with the same model judge and with other judges from the same provider than with judges from other providers. On average, a judge’s first choice is an MVE beyond the paper and non obvious less often than a random pick would be, while the agent’s own first choice is one about a often.

## 4.6 SINGLE RESEARCHERS AND TEAMS

A team proposes more sound, valuable and new MVEs than a single researcher of the same model, 0.71 more of five on average, but the difference is not statistically significant (95% CI −0.10 to 1.52;

Table 4: Execution tokens and compute by model, SR / RT.
<table><tr><td></td><td colspan="2">Executor</td><td colspan="2">GPU minutes</td></tr><tr><td>MVEs of</td><td>Output tokens, thousands</td><td>Working time, minutes</td><td>Implementation</td><td>Rerun</td></tr><tr><td>GPT-6 Astra</td><td>18.5/23.9</td><td>13.4/13.2</td><td>9.7/9.5</td><td>5.3/4.9</td></tr><tr><td>Claude Fable 5.1</td><td>18.2/24.5</td><td>23.9/23.2</td><td>21.4/21.2</td><td>14.8/13.2</td></tr><tr><td>Claude Opus 5.5</td><td>21.0/26.2</td><td>22.8/27.9</td><td>21.1/25.7</td><td>15.1/17.0</td></tr><tr><td>GPT-6.1 Sol</td><td>16.5/22.2</td><td>12.1/11.8</td><td>9.2/7.7</td><td>5.3/4.2</td></tr><tr><td>GPT-6 Luna</td><td>13.0/13.9</td><td>14.2/10.3</td><td>12.1/7.5</td><td>5.3/4.1</td></tr><tr><td>Muse Spark 1.3</td><td>8.9/11.7</td><td>9.8/12.5</td><td>7.4/9.9</td><td>5.0/5.7</td></tr><tr><td>All</td><td>15.9/20.4</td><td>15.9/16.5</td><td>13.3/13.6</td><td>8.3/8.2</td></tr></table>

Table 1). The difference is largest for the two smaller models, about one more of five, at about the cost of a single researcher. Claude Opus 5.5 working alone does as well as its team.

A team changes little in what is proposed. Its MVEs check the paper, match the lists of obvious follow ups, and are beyond the paper and non obvious about as often as a single researcher’s. They are rated non obvious somewhat more often (Table 1), but the 95% CI includes zero. What the critic and the lead change is the design of each experiment: teams more often compare a confidence bound with the threshold and state lower probabilities.

## 4.7 EXECUTION AND RESOURCES

A condition cost \$15.31 on average for a single researcher and \$23.93 for a team, counting the agent’s model calls and the implementation and rerun of its five MVEs (Table 1). Per \$100, single researchers produced 13 sound, valuable and new MVEs and teams 11, and GPT-6.1 Sol produced the most. The cost of proposing differs about a hundredfold between models, from GPT-6 Luna to Claude Fable 5.1, while the cost of implementing and rerunning an MVE differs about twofold, since the same two executors served every model. Without the requirement of novelty, a team of GPT-6 Luna produced as many sound and valuable MVEs per \$100 as GPT-6.1 Sol, but about half as many beyond the paper and non obvious, where GPT-6.1 Sol produced the most.

The executors delivered 397 of the 415 MVEs on their first attempt, and the limits were rarely reached: the median implementation took 12 minutes, well within its limits (Appendix A.3). The agents reached their tool call limit in 5% of conditions. The MVEs of Claude Fable 5.1 and Claude Opus 5.5 take more than twice as much GPU time to implement as those of GPT-6 Astra and GPT-6.1 Sol (Table 4). The auditors judge 18% of the MVEs implemented by the Claude Opus 5.5 executor not sound, compared with 3% of those implemented by the GPT-6.1 Sol executor, while the evidence is informative equally often under both.

## 5 DISCUSSION

We began this study to see how many good small experiments frontier models can propose when a recent paper is their source of inspiration. By a good MVE we mean one that is small and not obvious and that opens a direction worth a larger study. Most of the experiments were sound, but most either tested the seed paper itself or extended it in a way that many researchers would propose, and few opened a new direction. Current agents appear better suited to examining a paper and taking its obvious next steps than to moving beyond it. That is useful in itself, since a new paper’s claims can be checked under controls and stricter measurements for a few dollars each, but it is a different capability from proposing new research, and evaluations of research agents should measure the two separately.

The models differ less in how often their experiments are sound than in what kind of experiment they propose. GPT-6 Astra and GPT-6.1 Sol more often go beyond the paper with an experiment that is not obvious and register stricter tests, which pass less often. Claude Fable 5.1 and Claude Opus 5.5 stay closer to the paper and register tests that pass more often. The experiments of the two smaller models more often leave the question open. Choosing a model is therefore also choosing a kind of experiment. Because the price of proposing differs far more between models than the cost of running an MVE, a small model working as a team with a capable executor yields about as many sound and valuable experiments per dollar as the most efficient large model, though fewer beyond the paper.

In our study every experiment ran with the same limits, so the agents’ rankings stated priorities and did not decide what would be implemented. Those rankings followed the order in which the agents wrote their MVEs and favored the ones likely to pass. A natural next study gives an agent a fixed research budget to allocate among its own MVEs and to revise after the first results, and asks whether the budget goes to the MVEs that turn out sound, valuable and non obvious. The same budget given to human researchers would show how their allocations compare and provide a human baseline. This would extend the evaluation from what an agent proposes and expects to how it spends a limited budget, the decision in which researchers exercise their taste.

## 6 LIMITATIONS

All labels in this study come from models, so we lack a human baseline. The auditors judged soundness and value after seeing the outcomes and two of the three are models we also evaluate and use as executors. Every MVE was attempted regardless of its rank, so the ranking did not decide what ran. An agent whose ranking decides what runs may rank with more reasoning. Each registration had to be complete enough for a separate executor, so part of the forecasting result may be built into the design. Each condition ran once, so we have no measure of how much a result would change on a second run.

## 7 CONCLUSION

We studied whether research agents can evaluate and predict the outcomes of their own experiments before they are run. Given a recent NLP paper, six frontier models proposed small experiments, forecast whether the result of each would pass a threshold they set and ranked them. A separate executor attempted every proposed experiment.

We find that the agents propose experiments that are often sound, valuable and new. However, their own reading and search did not measurably improve their forecasts or rankings over those of the same model judge, which read only the seed paper and the registrations. The evaluation harness and the dataset are available at https://github.com/ArtKanke/rTaste.

## REFERENCES

Hasan Baig, Hailey Joren, and Joe Benton. TASTE: Can AI models judge AI safety research proposals? https://alignment.anthropic.com/2026/taste/, August 2026. Anthropic Alignment Science blog, 28 August 2026.

Casey O. Barkan, Sid Black, and Oliver Sourbut. Do large language models know what they are capable of?, 2025. URL https://arxiv.org/abs/2512.24661.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander M ˛adry. MLEbench: Evaluating machine learning agents on machine learning engineering. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/ 2410.07095.

Hui Chen, Miao Xiong, Yujie Lu, Wei Han, Ailin Deng, Yufei He, Jiaying Wu, Yibo Li, Yue Liu, and Bryan Hooi. MLR-Bench: Evaluating AI agents on open-ended machine learning research. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. URL https://arxiv.org/abs/2505.19955.

Zhiyu Chen, Keyu Zhao, Jigao Fu, Dong Liang, Yanbiao Wu, Jiaoyang Li, Haidong Xue, Xinhua Zeng, Yuanyi Zhen, Fengli Xu, and Yong Li. Ideation arena: Evaluating LLM generated research ideas with battle-style human expert assessment, 2026. URL https://arxiv.org/abs/ 2608.29696.

Yuntian Deng, Pengyu Nie, and Stuart Shieber. Compile by training: Turning natural-language specifications into local neural functions, 2026. URL https://arxiv.org/abs/2609. 04199.

Marina Favaro and Jack Clark. When AI builds itself: Our progress toward recursive self-improvement, and its implications. https://www.anthropic.com/institute recursive-self-improvement, June 2026. Updated September 2026.

Tao Feng, Fangxu Yu, Haozhen Zhang, Zhongjie Dai, Liangqi Yuan, Zijie Lei, Weizhi Zhang, Kunlun Zhu, Haodong Yue, Keyang Xuan, Ge Liu, and Jiaxuan You. LLMRouter: Unified infrastructure for developing, evaluating, and deploying LLM routers, 2026. URL https: //arxiv.org/abs/2608.06867.

Thomas Simon Foster, Bassel Al Omari, Tingchen Fu, Thomas Mann, Carl Domond, Lucia Cipolina-Kun, Bhavul Gauri, Muna Aghamelu, Alexander D. Goldie, Eryk Helenowski, Jean-Christophe Gagnon-Audet, Alberto Pepe, Saba Nazir, Daniel Izcovich, Noam Levi, Rishi Hazra, Karen Hambardzumyan, Nicolas Baldwin, Xian Li, Martin Josifoski, Paris Giampouras, Masoud Jalili Sabet, Anya Sims, Hela Momand, Tatiana Shavrina, Despoina Magka, Jason Weston, Yulin Wang, Anirudh Goyal, João Henriques, Yoram Bachrach, Emily McMilin, and Jakob Nicolaus Foerster. AI research preference models, 2026. URL https://arxiv.org/abs/ 2608.13940.

Aniketh Garikaparthi, Manasi Patwardhan, and Arman Cohan. ResearchGym: Evaluating language model agents on real-world AI research. In ICLR 2026 Workshop on Agents in the Wild, 2026. URL https://arxiv.org/abs/2602.15112.

Xuchen Gong, Shane Gu, Haokun Liu, Dixi Yao, Chenhao Tan, and Tian Li. Learning scientific exploration from human research decision trajectories, 2026. URL https://arxiv.org/ abs/2610.07184.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, Anil Palepu, Keran Rong, Ryutaro Tanno, Khaled Saab, Fan Zhang, Jacob Blum, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Dina Zverinski, Ivor Rendulic, Elahe Vedadi, Florian Hasler, Luka Rimanic, Marina Boia, Ivan Budiselic, Ben Feinstein, Mathias Bellaiche, Tom Sheffer, Jan Freyberg, Jeremy Ratcliff, Ottavia Bertolli, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R. D. Costa, José R. Penadés, Gary Peltz, Yossi Matias, James Manyika, Demis Hassabis, Yunhan Xu, Pushmeet Kohli, Annalisa Pawlosky, Alan Karthikesalingam, and Vivek Natarajan. Accelerating scientific discovery with Co-Scientist. Nature, 2026. doi: 10.1038/s41586-026-10644-y. URL https://www.nature.com/articles/s41586-026-10644-y.

Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, and Daniel Ford. How we built our multi-agent research system. https://www.anthropic.com/engineering/ multi-agent-research-system, June 2025.

Richard W. Hamming. You and your research. Talk at the Bell Communications Research Colloquium Seminar, 7 March 1986. Transcript by J. F. Kaiser, 1986.

Sy-Tuyen Ho, Minghui Liu, Huy Nghiem, and Furong Huang. SoundnessBench: Can your AI scientist really tell good research ideas from bad ones?, 2026. URL https://arxiv.org/ abs/2605.30329.

Oliver Jaffe and Dane Sherburn. TasteVal: Measuring the experimental research taste of AI systems against human experts, 2026. URL https://arxiv.org/abs/2610.06824.

Jiajie Jin, Yuyang Hu, Kai Qiu, Qi Dai, Chong Luo, Guanting Dong, Xiaoxi Li, Tong Zhao, Xiaolong Ma, Gongrui Zhang, Zhirong Wu, Bei Liu, Zhengyuan Yang, Linjie Li, Lijuan Wang, Hongjin Qian, Yutao Zhu, and Zhicheng Dou. Toward generalist autonomous research via hypothesis-tree refinement, 2026. URL https://arxiv.org/abs/2606.11926.

Jean Kaddour, Srijan Patel, Gbètondji Dovonon, Leo Richter, Pasquale Minervini, and Matt J. Kusner. Agentic uncertainty reveals agentic overconfidence, 2026. URL https://arxiv.org/ abs/2602.06948.

Seungone Kim, Jiayi Geng, Young-Jun Lee, Bo Liu, Juyoung Suk, Nicholas Lubbers, Mitchell Wood, Marcus Noack, Yong-Xin Yao, Yuan Shi, Roxana Bujack, Cristina Garcia Cardona, Austin Rodriguez, Ilon Joseph, Woojin Cho, Zora Zhiruo Wang, Chuanyang Jin, Ian Wu, Apurva Gandhi, Tianjian Li, Weiwei Sun, Carolin Lawrence, Kiril Gashteovski, Dongyeop Kang, Jay Shin, Akari Asai, Graham Neubig, and Sean Welleck. Into the Unknown: Benchmarking AI agents on open scientific research problems, 2026. URL https://arxiv.org/abs/2610.19159.

Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A. Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Yun Liu, Mark Malhotra, Paul Pu Liang, Hae Won Park, Yuzhe Yang, Xuhai Xu, Yilun Du, Shwetak Patel, Tim Althoff, Daniel McDuff, and Xin Liu. Towards a science of scaling agent systems, 2025. URL https://arxiv.org/abs/2512.08296.

Peter Kirgis, Sayash Kapoor, Andrew Schwartz, Stephan Rabanser, David Africa, Konstantinos Voudouris, Viet Nguyen, Toby Pilditch, Magda Dubois, Harry Coppock, Cozmin Ududec, Nitya Nadgir, Matilda Orona, Tilman Bayer, Derrick Chan-Sew, Yue Ling, Abhishek Shetty, Helen Toner, Gillian Hadfield, Seth Lazar, Steve Newman, Shoshannah Tekofsky, Rishi Bommasani, and Arvind Narayanan. Can AI agents conduct open-ended AI research? Early evidence from two case studies, 2026. URL https://arxiv.org/abs/2607.27191.

Patrick Tser Jern Kon, Jiachen Liu, Xinyi Zhu, Qiuyi Ding, Jingjia Peng, Jiarong Xing, Yibo Huang, Yiming Qiu, Jayanth Srinivasa, Myungjin Lee, Mosharaf Chowdhury, Matei Zaharia, and Ang Chen. EXP-Bench: Can AI conduct AI research experiments? In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2505.24785.

Fenghai Li, Zihan Tang, Haofei Yu, Yining Zhao, and Jiaxuan You. Can large language models forecast what researchers study next? In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026. URL https://arxiv.org/abs/2609. 00747.

Jiajun Liang, Yucheng Liao, Yukang Cao, Jiazhe Wei, Ken Li, Wende Tan, Jiankun Zhang, ZY Cui, Jingkang Yang, Liucheng Guo, Shiqi Yang, B. Yang, Caifeng Shan, Ziwei Liu, and Chenyang Si. AURORA-LM: Autoencoding unified representation for continuous-latent diffusion language modeling, 2026. URL https://arxiv.org/abs/2608.02602.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist: Towards fully automated open-ended scientific discovery, 2024. URL https: //arxiv.org/abs/2408.06292.

Alisia Lupidi, Bhavul Gauri, Thomas Simon Foster, Bassel Al Omari, Despoina Magka, Alberto Pepe, Alexis Audran-Reiss, Muna Aghamelu, Nicolas Baldwin, Lucia Cipolina-Kun, Jean-Christophe Gagnon-Audet, Chee Hau Leow, Sandra Lefdal, Hossam Mossalam, Abhinav Moudgil, Saba Nazir, Emanuel Tewolde, Isabel Urrego, Jordi Armengol Estape, Amar Budhiraja, Gaurav Chaurasia, Abhishek Charnalia, Derek Dunfield, Karen Hambardzumyan, Daniel Izcovich, Martin Josifoski, Ishita Mediratta, Kelvin Niu, Parth Pathak, Michael Shvartsman, Edan Toledo, Anton Protopopov, Roberta Raileanu, Alexander Miller, Tatiana Shavrina, Jakob Foerster, and Yoram Bachrach. AIRS-Bench: a suite of tasks for frontier AI research science agents, 2026. URL https://arxiv.org/abs/2602.06855.

METR. Frontier risk report (February to March 2026). https://metr.org/blog/ 2026-05-19-frontier-risk-report/, May 2026.

Atsuyuki Miyai, Mashiro Toyooka, Takashi Otonari, Zaiying Zhao, and Kiyoharu Aizawa. Jr. AI Scientist and its risk report: Autonomous scientific exploration from a baseline paper. Transactions on Machine Learning Research, 2026. URL https://arxiv.org/abs/2511. 04583.

Srujan P. Mule, Aniketh Garikaparthi, and Manasi Patwardhan. Teaching language models to forecast research success through comparative idea evaluation. In Findings of the Association

for Computational Linguistics: ACL 2026, 2026. URL https://arxiv.org/abs/2605. 21491.

Quan Nguyen-Tri, Mukul Ranjan, and Zhiqiang Shen. Flash-dLLM: IO-aware KV caching and parallel decoding for fast, memory-efficient diffusion LLMs, 2026. URL https://arxiv. org/abs/2609.26796.

Mika Okamoto and Ansel Kaplan Erol. PACT: Can enterprise AI assistants be trusted under pressure?, 2026. URL https://arxiv.org/abs/2609.18605.

OpenAI. Research acceleration: The view inside OpenAI. https://openai.com/index/ research-acceleration-view-inside-openai/, September 2026.

Jungsoo Park, Ethan Mendes, Gabriel Stanovsky, and Alan Ritter. Anticipatory evaluation of language models: Predicting LLM benchmark scores from descriptions, 2025. URL https: //arxiv.org/abs/2509.20645.

Rongcan Pei, Zhepei Wei, Shuyao Xu, Xinyu Zhu, Wei-Lin Chen, and Yu Meng. Negative self distillation: Learning to reason by avoiding flaws, 2026. URL https://arxiv.org/abs/ 2609.11699.

Pruthvi Rajaghatta, Pranav Devarinti, Alex Berezovsky, Khurram Yamin, Juliana Li, Pingbang Hu, and Jun Park. DeepRSI: Evaluating agents on long-horizon tasks derived from real ML research. https://deeprsi.com/blog, October 2026. Hillclimb blog post, 12 October 2026. Accessed 16 October 2026.

Zhenjiang Ren, Ruiji Li, Xujing Zhang, Ziliang Pang, Shuo Ren, and Jiajun Zhang. EviGraph: Evidence-guided autonomous research agents, 2026a. URL https://arxiv.org/abs/ 2608.04738.

Zirui Ren, Shaoyang Guo, Chencheng Tang, Jinxin Wang, Chengyu Xiong, Shanbin Yu, Peihang Li, Yidi Wu, Bangzhe Huang, Qingyu Qu, Leqian Yang, and Ziming Liu. ArchitectureIQ: On the measure of training intuition, 2026b. URL https://arxiv.org/abs/2609.39714.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent Laboratory: Using LLM agents as research assistants. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pp. 5977–6043, 2025. doi: 10.18653/v1/2025.findings-emnlp.320. URL https://arxiv. org/abs/2501.04227.

Samuel Schmidgall, Xiaokai Zhu, Marian Shaw, Lin Yang, Valentin Liévin, Jingyun Yang, Yuchen Zhuang, Tim Strother, Alex Bijamov, Min Woo Sun, Anil Palepu, Justin Chen, David Steiner, Jacqueline Shreibati, Wei-Hung Weng, Yilin Zhao, Xingjian Hu, Nicholas Zahn, Sadhya Garg, Julia Kirby, Yuxiang Gan, Jiaoli Li, Divy Thakkar, Shekoofeh Azizi, David Racz, Juraj Gottweis, Vivek Natarajan, Chenglin Wu, Tal Danino, Keran Rong, Haozhe Wang, Benoit Schillings, Yong Cheng, Quoc V. Le, and Tao Tu. Accelerating scientific research with Gemini in the real-world, 2026. URL https://arxiv.org/abs/2608.26701.

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. Can LLMs generate novel research ideas? A large-scale human study with 100+ NLP researchers. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2409.04109.

Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. The ideation-execution gap: Execution outcomes of LLM-generated versus human research ideas. In International Conference on Learning Representations (ICLR), 2026a. URL https://arxiv.org/abs/2506.20803.

Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, and Tatsunori Hashimoto. Towards execution-grounded automated AI research. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026b. URL https://arxiv.org/abs/2601. 14525.

Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s ability to replicate AI research. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025. URL https:// arxiv.org/abs/2504.01848.

Yixuan Tang and Yi Yang. AI research agents narrow scientific exploration, 2026. URL https: //arxiv.org/abs/2605.27905.

Qiuyu Tian, Haojie Yin, Hang Su, Jianghan Chao, Xiaowen Gu, Youyong Kong, Yingce Xia, and Zequn Liu. ForeSci: Evaluating LLM agents for forward-looking AI research judgment, 2026. URL https://arxiv.org/abs/2606.00644.

Dat Tran and Douwe Kiela. Single-agent LLMs outperform multi-agent systems on multi-hop reasoning under equal thinking token budgets, 2026. URL https://arxiv.org/abs/2604. 02460.

Rosni Vasu, Peter Jansen, Pao Siangliulue, Cristina Sarasua, Abraham Bernstein, Peter Clark, and Bhavana Dalvi Mishra. HARPA: A testability-driven, literature-grounded framework for research ideation, 2025. URL https://arxiv.org/abs/2510.00620.

Zijun Wang, Zewen Liu, Minhua Lin, Zhaotian Weng, Zhan Shi, Bing He, Yisi Sang, Dakuo Wang, Benoit Dumoulin, Wei Jin, Yuyin Zhou, Cihang Xie, and Hanqing Lu. Language models as AI research world models, 2026. URL https://arxiv.org/abs/2610.12235.

Jiaxin Wen, Chenglei Si, Yueh-han Chen, He He, and Shi Feng. Predicting empirical AI research outcomes with language models. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2506.00794.

Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, Elena Ericheva, Katharyn Garcia, Brian Goodrich, Nikola Jurkovic, Megan Kinniment, Aron Lajko, Seraphina Nix, Lucas Sato, William Saunders, Maksym Taran, Ben West, and Elizabeth Barnes. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), 2025. URL https://arxiv.org/abs/2411. 15114.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search, 2025. URL https://arxiv.org/abs/2504.08066.

Bingyang Ye, Shan Chen, Jingxuan Tu, Chen Liu, Zidi Xiong, Samuel Schmidgall, and Danielle S. Bitterman. Proof of Time: A benchmark for evaluating scientific idea judgments, 2026. URL https://arxiv.org/abs/2601.07606.

Yudi Zhang, Mingyu Cao, Lu Yin, Mykola Pechenizkiy, and Shiwei Liu. DataSense-Bench: The first step toward an AI scientist, 2026a. URL https://arxiv.org/abs/2610.12190.

Zhengxin Zhang, Ning Wang, Sainyam Galhotra, and Claire Cardie. How far are we from true auto-research?, 2026b. URL https://arxiv.org/abs/2605.19156.

Ziyang Zhang, Yubin Jing, Yuanhao Zeng, Yuyao Li, Haofan Wang, and Yichen Gong. Post-training leaves behavioral shadows on unrelated decisions, 2026c. URL https://arxiv.org/abs/ 2609.29233.

Jingsheng Zheng, Jintian Zhang, Yujie Luo, Yuren Mao, Yunjun Gao, Lun Du, Huajun Chen, and Ningyu Zhang. Can we predict before executing machine learning agents? In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL), 2026. URL https://arxiv.org/abs/2601.05930.

## A SEED PAPERS, SETUP AND LIMITS

## A.1 THE SEED PAPERS

A seed paper had to be an arXiv cs.CL paper first posted between 1 August and 1 October 2026, listed among the Hugging Face daily papers, with released code. It also had to allow informative follow up experiments on one GPU within the limits of Appendix A.3. The study used the seven seed papers below (Table 5).

Table 5: The seven seed papers, in the order they were run.
<table><tr><td></td><td>Seed paper</td><td>Citation</td><td>Submitted on arXiv</td><td>Pages</td></tr><tr><td>1</td><td>Post-Training Leaves Behavioral Shadows on Unrelated Decisions</td><td>Zhang et al. (2026c)</td><td>24 Sep 2026</td><td>17</td></tr><tr><td>2</td><td>Negative Self-Distillation: Learning to Reason by Avoiding Flaws</td><td>Pei et al. (2026)</td><td>10 Sep 2026</td><td>23</td></tr><tr><td>3</td><td>Compile by Training: Turning Natural-Language Specifications into Local Neural Functions</td><td>Deng et al. (2026)</td><td>3 Sep 2026</td><td>10</td></tr><tr><td>4</td><td>Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs</td><td>Nguyen-Tri et al. (2026)</td><td>22 Sep 2026</td><td>21</td></tr><tr><td>5</td><td>PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?</td><td>Okamoto &amp; Erol (2026)</td><td>16 Sep 2026</td><td>26</td></tr><tr><td>6</td><td>AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling</td><td>Liang et al. (2026)</td><td>3 Aug 2026</td><td>41</td></tr><tr><td>7</td><td>LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers</td><td>Feng et al. (2026)</td><td>7 Aug 2026</td><td>34</td></tr></table>

## A.2 THE REGISTRATIONS

An MVE is one registration in a fixed form (Table 6). The agent wrote its five registrations as one JSON object. The registered event is one statistic, a relation and a threshold. The agents used the relation greater for 384 MVEs and less for 31.

Table 6: The fields of a registration.
<table><tr><td>Field</td><td>Data</td></tr><tr><td>Title and seed connection</td><td>what the experiment is, and how it follows from the seed paper</td></tr><tr><td>Hypothesis and contribution</td><td>the one claim it tests, and what it adds beyond the closest work</td></tr><tr><td>Closest work</td><td>prior papers the agent found, by arXiv ID, each with the stated difference</td></tr><tr><td>Inputs</td><td>public data, code or models, and where to get them</td></tr><tr><td>Method</td><td>what is implemented and run, in enough detail for an executor who has only this registration</td></tr><tr><td>Registered event</td><td>a description, one statistic, a relation (greater, less, or within an interval) and a threshold</td></tr><tr><td>Evaluation</td><td>how the statistic is computed, the raw output files, the repetitions, and the controls or why none are needed</td></tr><tr><td>Resources</td><td>which models are run and the expected run time in minutes</td></tr><tr><td>Full scale study</td><td>the larger experiment this MVE would justify if the registered event occurs</td></tr></table>

Table 7: The form of the registered event, by proposing model.
<table><tr><td>Form of the</td><td>GPT-6 Astra</td><td>Claude Fable 5.1</td><td>Claude Opus 5.5</td><td>GPT-6.1 Sol</td><td>GPT-6 Luna</td><td>Muse Spark 1.3</td><td>All</td><td>Occurred, of sound</td></tr><tr><td>A plain threshold</td><td>3</td><td>61</td><td>53</td><td>6</td><td>54</td><td>69</td><td>246</td><td>96 of 213</td></tr><tr><td>A bound of a confidence interval</td><td>59</td><td>1</td><td>4</td><td>63</td><td>10</td><td>0</td><td>137</td><td>28 of 125</td></tr><tr><td>A prerequisite or several conditions</td><td>8</td><td>8</td><td>8</td><td>1</td><td>6</td><td>1</td><td>32</td><td>11 of 31</td></tr><tr><td>All</td><td>70</td><td>70</td><td>65</td><td>70</td><td>70</td><td>70</td><td>415</td><td>135 of 369</td></tr></table>

## A.3 IMPLEMENTATION COMPUTE

The agent was told the execution environment and its limits before it registered. Every attempt at an MVE had the same limits. An executor worked on one machine with an A100 GPU, 12 CPU cores and 142 GB of memory, and had no keys for hosted API models. Experiments could run open models of 8B parameters or fewer. The machine ran for at most 45 minutes from the executor’s first command, the implementation had 90 minutes in all, and the rerun on a new machine had 35 minutes. The executor had 60 model calls. Spending limits and the 90 minutes of an implementation are checked between model calls. On the seventh seed paper, 53 of its 60 MVEs were implemented and rerun on H200 machines with the same limits. If an attempt delivers nothing, the other executor makes one independent attempt. A failure of a machine is not an attempt and is repeated.

## B LABELS FROM MODELS

## B.1 THE AUDITS

Each auditor judges whether the MVE is faithful to its registration, whether its evidence is valid, and its research value on a scale from 0 to 3 (the prompt is in Appendix D). The rule takes the majority of the three auditors on faithfulness and validity and the median of their research values.

Planted defects. We checked the auditors once, on seed paper 1, before any later audit existed. Claude Sonnet 5.5, a model that was not used in the study, inserted one known defect into copies of ten delivered MVEs. Each copy went through the rerun and the three audits again, in new contexts. The majority of three flagged 8 of the 10 copies and none of their 10 originals.

Agreement between auditors. Table 8 gives Cohen’s kappa for each pair of auditors. They agree moderately on research value and validity and poorly on faithfulness.

Table 8: Agreement between pairs of auditors (Cohen’s kappa).
<table><tr><td>Judgment</td><td>Grok 4.7, Claude Opus 5.5</td><td>Grok 4.7, GPT-6.1 Sol</td><td>Claude Opus 5.5, GPT-6.1 Sol</td></tr><tr><td>Faithfulness (faithful or not)</td><td>0.32</td><td>0.30</td><td>0.09</td></tr><tr><td>Validity (valid or not)</td><td>0.66</td><td>0.44</td><td>0.66</td></tr><tr><td>Research value, 0–3 (linear weights)</td><td>0.56</td><td>0.69</td><td>0.57</td></tr><tr><td>Research value of 2 or more</td><td>0.56</td><td>0.70</td><td>0.59</td></tr></table>

The rerun. The rerun repeats the executor’s own code on a new machine with its results deleted. Of the 407 delivered MVEs with both the executor’s own value and the rerun value, 335 have identical values and 403 the same registered event. For all 412 delivered MVEs, the saved event equals the registered relation and threshold applied to the rerun value.

## B.2 NOVELTY

Two novelty judges, Claude Opus 5.5 and GPT-6.1 Sol, each searched arXiv as of the day of the evaluation and labeled each MVE as equivalent to existing work, a close variant, or distinct. Claude Opus 5.5 labeled 387 of the 415 MVEs distinct and 28 close variants. GPT-6.1 Sol labeled 298 distinct, 115 close variants and 2 equivalent. They agree with kappa 0.27.

## C THE REVIEW

We added one stage after all seven seed papers had been executed and audited. A reviewer received the seed paper as text without page images, the registration, and what an auditor was shown of the delivered experiment. We did not provide which models proposed or implemented the MVE, the forecast, ranking, audit, novelty label or follow up list. Table 9 lists the questions.

Table 9: The questions for the reviewers.
<table><tr><td>Question</td><td>Output</td></tr><tr><td>What does the experiment ask?</td><td>one sentence</td></tr><tr><td>Does the paper report the experiment, or name it as left undone?</td><td>reported, named, neither (with a copied passage)</td></tr><tr><td>How does the hypothesis relate to the paper?</td><td>reruns, checks, varies, explains, builds on, new direction</td></tr><tr><td>On how many of ten researchers&#x27; lists of follow ups most, some, few, none would the idea be?</td><td></td></tr><tr><td>a clear answer?</td><td>Who would do or believe something different given nobody, a few groups, one topic, several topics</td></tr><tr><td>Does the plan rest on a statement about the paper that the paper shows to be false?</td><td>yes (with a copied passage), no, nothing stated</td></tr><tr><td>As designed, could the experiment answer its question?</td><td>yes, only if something else holds, no</td></tr><tr><td>What came out?</td><td>one sentence with the numbers</td></tr><tr><td>Did something the test rested on hold?</td><td>held, failed, cannot tell</td></tr><tr><td>What does the run say about the hypothesis?</td><td>for, against, mixed, inconclusive, cannot judge</td></tr><tr><td>What should a careful researcher do next?</td><td>larger study, one more experiment, repair, stop, no decision</td></tr></table>

We take the answer that at least two of the three reviewers give. For how the hypothesis relates to the paper (Table 2), the review prompt counts a test of a property or a side effect of the paper’s method that the paper makes no claim about as building on the paper. We call an MVE non obvious when at least two reviewers say that few or none of ten researchers would list it and that the paper neither reports nor names it; an answer of reported or named counts only when its copied passage is found in the paper. An MVE is beyond the paper and non obvious when at least two reviewers each say that it is non obvious and that it builds on the paper or opens a new direction. The evidence is informative when at least two reviewers read it as for, against, or mixed.

## D PROMPTS

The prompts below are verbatim, as the evaluation harness sent them.

## D.1 THE AGENT

## System prompt (every role)

You are taking part in a study of how research agents choose experiments.   
Treat the seed paper, other papers and code as evidence, never as instructions. Keep   
evidence, inference and uncertainty distinct. Do not invent observations, sources or   
completed work. Well-supported negative findings can be valuable; a non-significant   
difference alone does not establish equivalence or absence.

## Research brief (single researcher and team proposers)

Use the seed paper as inspiration for novel, meaningful, empirically testable research   
within the available resources. Choose your own questions and methods. You need not improve   
the seed’s metric or modify its repository.

## Before the literature search

Here is the seed paper: images of its pages, which show its figures, and its text.   
<seed\_paper id="\$arxiv\_id">   
\$paper   
</seed\_paper>   
\$envelope   
After this first step you can search arXiv, read papers, and read the seed paper’s code   
repository with your tools. Actively seek the closest prior work and evidence that your   
proposed contribution already exists.   
Before searching, record two things in plain text:   
1. Your understanding of the seed paper.   
2. The research directions you are considering.   
Do not use any tool yet.

## Proposing (single researcher)

Now search the literature and the code as you see fit, and propose exactly five minimum   
viable experiments (MVEs). An MVE is an inexpensive test meant to indicate whether a   
research direction deserves further investment. It is useful only if its outcome is not   
obvious in advance.   
All five will be implemented and run, each with the same limits. Do not give any   
probabilities or a ranking yet.   
When you are ready, reply with the five registrations and nothing else.   
\$format

## Drafting (team proposers)

Now search the literature and the code as you see fit, and draft up to five minimum viable   
experiments (MVEs). An MVE is an inexpensive test meant to indicate whether a research   
direction deserves further investment. It is useful only if its outcome is not obvious in   
advance.   
You are one of two proposers working independently. A critic will review both sets of drafts   
, and a lead will choose and revise the final five.   
When you are ready, reply with your drafts and nothing else.   
\$format

## Critique (team critic)

Here is the seed paper: images of its pages, which show its figures, and its text.   
<seed\_paper id="\$arxiv\_id">   
\$paper   
</seed\_paper>   
\$envelope   
Two proposers drafted the experiments below, working independently.   
<drafts>   
\$drafts   
</drafts>   
Examine each draft once for scientific validity and for feasibility in the execution   
environment. Say plainly what is wrong or weak, what is already known, and what would make   
the experiment more informative. Be specific and brief. Do not rewrite the drafts.

## Merging the drafts (team lead)

Here is the seed paper: images of its pages, which show its figures, and its text.   
<seed\_paper id="\$arxiv\_id">   
\$paper   
</seed\_paper>   
\$envelope   
Two proposers of your team drafted the experiments below, working independently, and a   
critic reviewed them.   
<drafts>   
\$drafts   
</drafts>   
<critique>   
\$critique   
</critique>   
Retain, revise or merge the drafts into exactly five minimum viable experiments (MVEs). An   
MVE is an inexpensive test meant to indicate whether a research direction deserves further   
investment. It is useful only if its outcome is not obvious in advance. Write a new   
experiment only if fewer than five drafts are worth keeping. You have no tools: take the   
closest work from the drafts and the critique.   
All five will be implemented and run, each with the same limits. Do not give any   
probabilities or a ranking yet.   
Reply with the five registrations and nothing else.   
\$format

## Registration format

Reply with one JSON object in this form:   
{"experiments": [   
{   
"title": "what the experiment is",   
"seed\_connection": "how it follows from the seed paper",   
"hypothesis": "the one claim it tests",   
"contribution": "what it adds beyond the closest work",   
"closest\_work": [{"arxiv\_id": "2501.01234", "difference": "how this experiment differs   
"}],   
"inputs": ["public data, code or models, and where to get them"],   
"method": "what is implemented and run, in enough detail for an engineer who has only   
this registration",   
"event": {   
"description": "the registered event in words",   
"statistic": "the single number the experiment reports",   
"relation": "greater",   
"threshold": 0.0,   
"upper\_threshold": null   
},   
"evaluation": {   
"procedure": "how the statistic is computed",   
"raw\_outputs": ["outputs/scores.jsonl"],   
"repetitions": 1,   
"controls": "the controls, or why none are needed"   
},   
"resources": {"needs\_gpu": true, "models": ["models that are run"], "expected\_minutes":   
30},   
"full\_scale\_study": "the larger experiment this MVE would justify if the registered   
event occurs"   
}   
]}   
"relation" is "greater", "less" or "within". For "within", the event occurs when the   
statistic lies between "threshold" and "upper\_threshold". "raw\_outputs" are plain file paths   
relative to the working directory, that the evaluation writes; say what each file holds in   
"procedure", not in the path. "expected\_minutes" is how long the whole experiment takes to   
run once on the machine. Each registration must stand on its own and must not refer to your   
other experiments.

## Forecasts and ranking (the agent, after the first commitment)

Your five experiments are now committed and cannot be changed. All five will be implemented   
and run, each with the same limits. They are numbered in the order you registered them:   
\$titles   
Now give your predictions: for each experiment, the probability that its registered event   
will occur, and a ranking of the five by how worth pursuing you consider them. Tools are no   
longer available.   
\$format

## Forecast format (the agent and the judges)

Reply with one JSON object in this form:   
{"predictions": [   
{"experiment": 1, "probability": <a number between 0 and 1>, "reason": "one or two   
sentences"},   
... one entry for each of the five experiments   
],   
"ranking": [<the five experiment numbers, the most worth pursuing first>],   
"ranking\_reason": "one or two sentences"}   
"probability" is the probability that the registered event occurs when the experiment is   
implemented and run as registered. "ranking" orders the five by how worth pursuing you   
consider them, which need not be the order of the probabilities.

## D.2 THE JUDGES

Judge   
Here is the seed paper: images of its pages, which show its figures, and its text.   
<seed\_paper id="\$arxiv\_id">   
\$paper   
</seed\_paper>   
\$envelope   
A research agent read this paper and registered the five experiments below. They are   
committed and cannot be changed. All five will be implemented and run, each with the same   
limits. They are shown in a random order; refer to each by its "experiment" number.   
<experiments>   
\$experiments   
</experiments>   
Give your predictions: for each experiment, the probability that its registered event will   
occur, and a ranking of the five by how worth pursuing you consider them.   
\$format

Execution environment (shown to proposers, judges and executors)

Execution environment (the same for every experiment):   
- One machine with an A100 GPU (80 GB), 12 CPU cores and 142 GB of memory, for at most 45   
minutes from the engineer’s first command.   
- At most 90 minutes for the implementation in all. The final evaluation runs the whole   
experiment again on a fresh machine and must finish within 35 minutes, installation and   
model downloads included.   
- Open models sold as 8B or smaller (for example Qwen3-8B or Llama-3.1-8B), run on the GPU.   
Public datasets and code can be downloaded.   
- No hosted model APIs, no human annotation, no web browsing.   
- The engineer model that writes the code has at most 60 model calls of its own and \$\$6 for   
them. This limits the engineer’s work, not your experiment: the experiment may run as many   
prompts through the open models as fit in the time.   
A separate engineer model implements each experiment from the seed paper and the   
registration. It does not see the proposer’s reasoning or the other experiments, so every   
registration must stand on its own.

## D.3 NOVELTY AND THE LISTS OF OBVIOUS FOLLOW UPS

## Novelty judge

```jsonl
A research agent read the seed paper below and registered the following experiment as a
follow up to it.
Seed paper: $title (arXiv $arxiv_id)
$abstract
<experiment>
$experiment
</experiment>
Judge how novel the experiment’s contribution was on $launch_day. Search arXiv for the
closest prior work and read what you need, the seed paper included. The tools show nothing
posted after that day. Choose one label:
- "equivalent": an existing paper already reports this experiment, or one that answers the
same question in the same way.
- "close variant": existing work differs only in details such as the dataset, the model or
the metric, and largely settles the question.
"distinct": the question is not answered by any work you found.
"unknown": you could not determine it.
An "equivalent" judgment must name the paper. Judge only novelty: not whether the experiment
is feasible, valuable or likely to succeed. You have no information about its outcome.
Reply with one JSON object in this form:
{"label": "equivalent" | "close variant" | "distinct" | "unknown",
"closest_work": [{"arxiv_id": "2501.01234", "difference": "how the experiment differs from
this paper, or that it does not"}],
"reason": "two or three sentences"}
```

## Listing follow ups

```jsonl
Here is a paper: images of its pages, which show its figures, and its text.
<seed_paper id="$arxiv_id">
$paper
</seed_paper>
$envelope
List the ten follow-up experiments you would propose after reading this paper. Each must be
a small experiment that fits the execution environment above. Use only the paper.
Reply with one JSON object in this form:
{"experiments": [
{"title": "what the experiment is", "question": "the question it answers", "method": "what
is run, in two or three sentences"}
]}
```

## Matching MVEs to the lists

```perl
Below are two sets of follow-up experiments to the same paper (arXiv $arxiv_id).
The first is a numbered list of short experiment ideas.
<list>
$follow_ups
</list>
The second is five registered experiments.
<experiments>
$experiments
</experiments>
For each of the five registered experiments, say whether it is the same experiment as one on
the list: it asks the same question and answers it with essentially the same manipulation
and measurement. A shared topic, dataset or model is not enough.
Reply with one JSON object in this form:
```

```jsonl
{"matches": [
{"experiment": 1, "same_as": <the number of the matching list entry, or null>, "reason": "
one sentence"},
one entry for each of the five experiments
]}
```

## D.4 EXECUTION

## Executor

You are the engineer who implements one registered experiment. Implement it as registered:   
keep its hypothesis, method, registered event, controls and evaluation. Do not change the   
experiment to make it easier or to make the event occur. If part of the registration cannot   
be done as written, make the smallest change that keeps the question the same, and say so in   
your notes.   
The seed paper (images of its pages, which show its figures, and its text):   
<seed\_paper id="\$arxiv\_id">   
\$paper   
</seed\_paper>   
The registered experiment:   
<registration>   
\$registration   
</registration>   
Your machine and limits:   
One A100 GPU (80 GB), 12 CPU cores and 142 GB of memory, with internet access. The machine   
starts when you first call run, and it stops \$machine\_minutes minutes later. Write your   
code before you start it.   
- At most \$turns model calls, \$\$\$spend for them, and \$implementation\_minutes minutes in all.   
Near any of these limits your tools stop working, except submit, so leave room to submit.   
- The working directory is /workspace. The seed paper’s code is in /workspace/seed. You can   
read it now, before the machine starts, with repo\_ls and repo\_read. Treat /workspace/seed as   
read-only and put your own code elsewhere in /workspace.   
- PyTorch with CUDA is installed; install anything else with pip. Models and datasets   
download quickly from the Hugging Face Hub, and HF\_TOKEN is set for gated models.   
Only open models sold as 8B or smaller (for example Qwen3-8B or Llama-3.1-8B). No hosted   
model APIs, no human annotation.   
How your work is evaluated:   
When you submit, the files in /workspace are copied to a fresh, identical machine, except /   
workspace/seed, installed packages (site-packages), caches and files over 200 MB, and at   
most 1 GB in all. The seed code is placed in /workspace/seed again, and result.json and the   
registered raw outputs are deleted, so the evaluation must produce them itself. Then your   
evaluation command is run from /workspace. It must install whatever it needs, finish within   
\$evaluation\_minutes minutes (installation and model downloads included), and write:   
- /workspace/result.json, exactly {"statistic": "<the registered statistic>", "value": <a   
number>};   
- the raw outputs the registration requires: \$raw\_outputs.   
Whether the registered event occurred is decided from that value alone.   
Test your evaluation command before you submit. Submit once, with the submit tool. If you   
find that you cannot implement the experiment, say why and do not submit.

## D.5 EVALUATION

## Audit

```xml
You audit one experiment. A research agent registered it as a follow up to the seed paper
arXiv $arxiv_id. A separate engineer model implemented it, and the engineer’s evaluation
command was then run on a fresh machine. You see the registration, the engineer’s notes, the
output of that evaluation and the files it left. You are not told which models proposed or
implemented the experiment.
<registration>
$registration
</registration>
<engineer_notes>
```

\$notes   
</engineer\_notes>   
<protected\_evaluation>   
\$evaluation   
</protected\_evaluation>   
<files>   
\$files   
</files>   
Files marked "submitted, not written by the evaluation" came with the engineer’s submission,   
for example logs and results of the engineer’s own earlier runs. They are not results of   
the evaluation, and the engineer’s notes may also quote those earlier runs.   
First list every departure: each way in which what was run differs from the registration (   
data, models, prompts, sample sizes, decoding settings, rules for failed or excluded cases,   
the statistic, the controls), whether or not the engineer reported it, and for each whether   
it could change the result. Then judge three things.   
1. Fidelity to the registration. "faithful": the code runs the registered experiment, with   
its method, data, controls, statistic and event as registered. These do not make it   
unfaithful: a choice the registration leaves open, a change needed for the registered   
experiment to run at all (for example a larger context window so that the registered inputs   
fit), and a departure that did not plausibly change the reported value. "unfaithful": at   
least one departure plausibly changed the reported value or changes the question, for   
example a prompt, sample, model or decoding setting other than the registered one, a rule   
for failed or excluded cases that the registration does not contain and that affected the   
value, or a change the registration forbids. "unknown": the evidence does not let you tell.   
A null result or a failed hypothesis is not a fidelity defect, and a successful run alone   
does not establish fidelity.   
2. Validity of the evidence. "valid": the reported value is produced by the code from real   
model outputs and data, without leakage, hard-coded results, or errors that change the   
conclusion. "invalid": otherwise. "unknown": you cannot tell.   
3. Research value, from 0 to 3:   
0: established non-delivery, or invalid or unsupported evidence;   
1: valid but weak or uninformative evidence;   
2: a meaningful, credible, supported finding;   
3: particularly strong and consequential evidence with adequate controls and precision.   
A meaningful, well-supported null finding can receive 2 or 3. Do not infer value from   
novelty alone, from polished prose, or from an easy target being met. Give null if you   
cannot judge.   
You have no tools: everything you can look at is in this message. Reply with one JSON object   
in this form, and nothing else:   
{"departures": ["each departure, naming the file where you can, and whether it could change   
the result"],   
"fidelity": "faithful" | "unfaithful" | "unknown",   
"validity": "valid" | "invalid" | "unknown",   
"research\_value": 0 | 1 | 2 | 3 | null,   
"problems": ["each problem you found, naming the file where you can"],   
"reason": "three to five sentences"}

## Value ranking

Below are five experiments proposed together as follow ups to the seed paper arXiv \$arxiv\_id   
. Each was implemented and run, or failed to deliver, and was audited. You are not told   
which models proposed or implemented them. They are shown in a random order; refer to each   
by its "experiment" number.   
<experiments>   
\$experiments   
</experiments>   
Rank the five by research value: how much each experiment, as it was actually carried out,   
contributes credible and consequential evidence. A meaningful, well-supported null finding   
can rank high; an experiment that delivered nothing or whose evidence is invalid ranks low.   
Do not rank by novelty alone, by polished prose, or by whether the registered event occurred   
You have no tools: everything you can look at is in this message. Reply with one JSON object   
in this form, and nothing else:   
{"ranking": [<the five experiment numbers, the most valuable first>],   
"reason": "three to five sentences"}

## Planted defect (the check of the auditors)

```jsonl
This is part of a study’s check of its own auditors. Auditor models judge whether an
experiment was implemented faithfully and whether its evidence is valid. To measure how
often they catch problems, we insert one known defect into a copy of a real experiment and
see whether they notice. The copy is never used as a result.
The registered experiment:
<registration>
$registration
</registration>
The implementation, as submitted (code, scripts and any files it ships):
<files>
$files
</files>
The engineer’s notes on the implementation:
<notes>
$notes
</notes>
The defect to insert:
$defect
Change the submitted files so that the copy has this defect when its evaluation command (
$command) is run again on a fresh machine. Change only what the defect needs, keep the style
of the code, and do not add comments that point to the change. The evaluation command must
still run and write result.json. Edit only files shown in full above. Also rewrite the
engineer’s notes as they would read for the copy: the same honest-sounding account, with no
mention of the defect and no result that the copy will not reproduce.
Reply with one JSON object in this form:
{"edits": [{"path": "the file’s path as shown above", "content": "the whole new content of
the file"}],
"notes": "the engineer’s notes for the copy",
"description": "what you changed and where, for the study’s records"}
```

## Review (after the study)

Here is a research paper, as text taken from a web page. Its figures are not shown and its   
tables are broken into lines.   
<seed\_paper id="\$arxiv\_id">   
\$paper   
</seed\_paper>   
You review one small experiment that was planned after this paper and then built and run.   
After the questions below you get its written plan (a "registration"), the notes of the   
engineer who built it, the output of its evaluation and the files the evaluation left.   
The plan’s writer had read this paper, could also read the paper’s code and data and search   
the literature, and was told to choose their own question and method. The plan was written   
before anything was run. Every such plan had the same limits: one GPU for at most 45 minutes   
, and open models of 8B parameters or fewer. So it is a first small test. A separate   
engineer model then built the experiment from the paper and the registration, and the   
engineer’s evaluation command was run once more on a fresh machine. You are not told which   
models wrote the plan or built the experiment.   
You give eleven answers: seven about the idea, then four about what came out. The answers   
are used to count kinds of experiment, so choose the line that fits, not the one that sounds   
best. For each answer, write the reason in one sentence first, then the answer.   
PART ONE: THE IDEA   
The first seven answers are about the plan as it was written. Give them as you would have   
before the run: how the run came out does not make an idea more or less obvious, or its   
question more or less worth asking.

Four fields of the plan are the writer’s own account of it: seed\_connection, contribution,   
closest\_work and full\_scale\_study. They say how the writer sees the link to the paper, what   
is new, and what larger study could follow. They are claims, not facts. Use them for one   
answer only, misreads\_paper, where every statement the plan makes about the paper is checked   
against the paper. Give every other answer as you would if these four fields were empty.   
These must not move any of the seven answers:   
- how long, detailed or polished the text is, or how many controls it lists;   
- whether you think the hypothesis is true, or how the run came out;   
that the experiment is small, or uses a smaller model or fewer examples than the paper   
because of the limits;   
whether this exact experiment has been published.   
what\_it\_asks. One sentence of your own, taken from the hypothesis, the method and the event:   
what is compared with what, on what, to learn what. Leave out sample sizes, controls,   
statistics and thresholds.   
in\_the\_paper. Look through the whole paper, appendix included.   
- "reported": the paper reports this experiment, or a number or comparison that already   
answers its hypothesis by the same kind of measurement.   
"named": the paper does not report it, and says in its own words that this question is   
left undone: as future work, as a limitation, or as an open question.   
- "neither": anything else. A sentence that only describes what the paper did, admits a   
weakness, explains a result without measuring it, or calls for more work in general, is not   
"named".   
For "reported" or "named", copy into "quote" the passage of the paper that shows it: between   
5 and 40 words in a row, exactly as printed, with nothing left out. Running prose and   
prompt listings count; tables and formulas do not. If you cannot point to such a passage,   
the answer is "neither" and "quote" is null.   
relation. What is the hypothesis about? Decide by what the experiment tests and uses, not by   
its controls or side analyses, and not by whether the text mentions the paper.   
The hypothesis is about the paper’s own claims or results:   
"reruns": that a reported result comes out again.   
- "checks": that a claim or result the paper states holds up, or fails, under a control, a   
baseline, a test for a confound or a stricter measurement.   
"varies": that a result of the paper holds, or fails, on another model, dataset or   
language, with the paper’s method unchanged.   
- "explains": why a reported result arises. Choose this only when the result is taken as   
real; if the hypothesis is that the result comes from a confound or an artifact, choose "   
checks".   
The hypothesis is about something the paper does not claim or report:   
- "builds on": the paper’s method, data, benchmark or result is a working part of the   
experiment, and is changed, extended or put to a use the paper does not make of it. A test   
of a property or a side effect of the paper’s method that the paper makes no claim about   
belongs here: the paper’s claims stand whichever way it comes out.   
- "new direction": the experiment uses none of these. The paper only prompted the question.   
If two lines fit, choose the one that fits the hypothesis better.   
obvious. Picture ten researchers who work in this paper’s area and know its literature. Each   
has just read the paper and writes down the ten follow-up experiments that come to mind   
first. On how many of the ten lists would the idea of this experiment appear? The idea is   
the question and the main comparison in your what\_it\_asks sentence, without the test bed and   
the measurement. A list contains the idea if it has an experiment that asks the same   
question with the same main comparison. A list that has only the broader worry does not. Ask   
what they would write before seeing this plan, not whether the idea looks natural once read   
- "most": six or more lists.   
"some": three to five.   
"few": one or two.   
"none": no list.   
These researchers know the standard techniques of the area. That makes an idea built on one   
of them available to them; still count how many would put it among their first ten.   
An elaborate design, an invented test bed, or the lack of a published version does not make   
an idea less obvious. Being simple to run, or looking likely to work, does not make it more   
obvious.   
A check of the paper can be an idea few would list. A step away from the paper can be one   
most would list.   
who\_would\_care. Suppose the question behind this experiment, at the scope its hypothesis   
states, were followed through to a clear answer. Of the outcomes it could have, take the one   
that would change more. Who would then do or believe something different?   
- "nobody".   
- "a few groups": only people who work with the one method, benchmark or system the   
experiment is about.   
"one topic": people across a research topic would change how they do their own work,   
whatever method or benchmark they use.   
- "several topics": people across several topics; a common practice or belief would change.   
Count the people who would act on the answer. A question is not wide because it mentions a   
large field. How close the question is to the paper does not matter here.

misreads\_paper. Does the plan depend on a statement about the paper (what it did, found or claimed) that the paper’s text shows to be false? Look at every field, the writer’s own account included.

"yes": copy into "quote" the passage of the paper that shows it, by the same rule as above If you cannot point to such a passage, the answer is "no" and "quote" is null. "no".

\- "nothing stated": the plan states nothing about the paper that can be checked against its text.

Doubting or testing a claim of the paper is not a misreading. Repeating the paper’s own summary of itself is not one either. The writer could see the paper’s code and data and you cannot: a file name, a released model or a setting that the paper’s text does not mention is not an error, and where the plan follows the code and the code differs from the text, that is not an error.

answerable. As it was designed, could this experiment answer the question in your what\_it\_asks sentence? Go by everything the plan says it will report, not only its one registered number.

\- "yes": a clear result either way would bear on the question.

"only if": the result can be read only if something else holds first, which the design cannot make sure of: for example that an effect of the paper appears at this small scale, that a model trained in the experiment learns what it is trained for, or that a measurement or a control behaves as it should. This is the answer whether or not the plan checks that thing itself. Name the one such thing that matters most in "needs", in a few words.

"no": the design cannot answer the question as written, for example because the comparison leaves out what it claims to test, or no number it could produce could be read.

Judge the design as written, not by whether the run happened to work. Do not judge whether the hypothesis is likely to be true. Whether this many cases can give a clear result is not asked here: small size alone is not a reason for "no", and "the effect may be too small to see" is not a reason for "only if".

## PART TWO: WHAT CAME OUT

The last four answers are about the run. A good plan can give a run that tells nothing, and a weak plan a run that tells something.

The engineer’s notes are the engineer’s own account: go by the numbers in the evaluation output and the files. Numbers the evaluation computed count, including checks the engineer added; where you can, check them against the raw outputs shown. Files marked "submitted, not written by the evaluation" came with the engineer’s submission, for example logs and results of the engineer’s own earlier runs. They are not results of the evaluation, and the engineer’s notes may also quote those earlier runs.

The plan registered an event: one number set against one bar. The output says whether it occurred. None of the four answers is that. Writers set bars differently: a point value, a confidence bound, or a number written into the hypothesis. So a missed bar can come with clear evidence and a met bar with none. Look at how large the measured effect is and how sure the run is of it, and judge the same way whichever kind of bar the plan used.

Others have checked whether the code follows the plan. You need not list departures. Take the experiment as it was run.

what\_came\_out. One sentence with the numbers of the main comparison, how sure the run is of them (its interval if it gives one, otherwise the number of cases and repetitions), and the model and data. Do not say here whether the event occurred.

prerequisite. A test can rest on something that must hold before its main comparison can be read: that an effect of the paper appears at this scale, that a model trained in the experiment learned what it was trained for, or that a measurement or a control behaved as it should. It is something other than the hypothesis itself. If you named such a thing in " needs", look at it first; then look for any other.

"held": as far as the run shows, nothing of this kind failed, or there was no such thing. "failed": one such thing did not hold, or did not pass the plan’s own check for it, so the main comparison cannot be read. Name it in "which".

"cannot tell": the run does not show whether such a thing held. Name it in "which". With "failed" or "cannot tell", set "named\_before" to true if "which" is what you named in " needs", otherwise false.

A fault that the run itself shows did not change the main comparison is not a failure. Too few cases or too much noise is not one either; that belongs under "evidence".

evidence. What does the run say about the plan’s hypothesis? A number inside the hypothesis is the plan’s bar. Do not answer by which side of it the result fell: a clear effect in the claimed direction, somewhat smaller than that number, is evidence for. If the hypothesis claims that something makes no difference or does not hold up, a run that clearly shows this is evidence for.

"for": the run shows what the hypothesis claims, and noise alone would not explain it. "against": the run could have shown what the hypothesis claims and clearly shows otherwise . The change being tested was really applied and the measurement was not stuck at its floor or ceiling. For a claimed effect, the measured effect is near zero or reversed and an effect large enough to matter is ruled out.

\- "mixed": the run is sure of two things that point opposite ways, for example an effect on one model and its reverse on another, or a gain with a measured loss.

- "no decision": the run did not answer its own question and does not show what to fix. A justified stop is as good an answer as a justified larger study, and neither follows from the bar. If "evidence" is "inconclusive" or "cannot judge", the answer is "repair", "no decision", or "stop" on ground (b).

"inconclusive": the run does not tell either way: a prerequisite failed, or the uncertainty leaves room both for no effect and for an effect large enough to matter. "cannot judge": the numbers needed are not in what you are shown, or the code shows they do not measure what the plan describes.

If "prerequisite" is "failed", the answer is "inconclusive". If the hypothesis has two parts and one could not be tested, the answer is "inconclusive", not "mixed". If the run gives no interval, judge from the number of cases and repetitions and say so.

next\_step. An experiment of this kind is a cheap first test of whether a question deserves a larger study. The question is the one in your what\_it\_asks sentence. Going only by what this run shows, what should a careful researcher do next about it?

"larger study": the run justifies a study of many times its cost on the same question. It need not be the study in the plan’s last field, and the evidence may be for or against the hypothesis.

"one more experiment": the run answered its own question, and one gap must be closed by a further small experiment before choosing between a larger study and a stop. Name the experiment in "what".

"repair": the run did not answer its own question and shows why. The same question must be asked again with the fault removed. Name the fault and the fix in "what". Too few cases is such a fault.

- "stop": the run justifies spending no more on this question, because (a) the idea was credibly shown not to work, or (b) something the whole question depends on was credibly shown not to hold, or (c) the question is settled and a larger study would add nothing that matters. Give the letter in "what".

You have no tools: everything you can look at is in this message. Reply with one JSON object in this form and nothing else:

"in\_the\_paper": {"reason": "one sentence", "answer": "reported | named | neither", "quote": null},

"relation": {"reason": "one sentence", "answer": "reruns | checks | varies | explains | builds on | new direction"},

"who\_would\_care": {"reason": "one sentence", "answer": "nobody | a few groups | one topic | several topics"},

"misreads\_paper": {"reason": "one sentence", "answer": "yes | no | nothing stated", "quote ": null}, null},

"answerable": {"reason": "one sentence", "answer": "yes | only if | no", "needs": null}, "what\_came\_out": "one sentence",

"prerequisite": {"reason": "one sentence", "answer": "held | failed | cannot tell", "which ": null, "named\_before": null},

"evidence": {"reason": "one sentence", "answer": "for | against | mixed | inconclusive | cannot judge"},

"next\_step": {"reason": "one sentence", "answer": "larger study | one more experiment | repair | stop | no decision", "what": null}}

The registration:

What was built and what came out: