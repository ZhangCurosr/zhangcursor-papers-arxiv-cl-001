# Swapping the Judge Changes the Ranking: A Preregistered Study of Cross-Provider Replication in LLM-Judged System Comparisons

Sangdo Han CEO Business School sangdohan@ceobizschool.kr

Moonsoo Kim CEO Business School mskim@ceobizschool.kr

## Abstract

Practitioners who compare systems with an LLM judge and have no human labels often check a conclusion by repeating it with a judge from another provider. We test how far that check can be trusted. In a preregistered study (with a disclosed post-registration reduction of the judge set), five judge models from three providers re-judged 4,000 Chatbot Arena battles (nine systems, 33 system pairs) that carry human votes, each in both response orders. Agreement with the human vote was low at the verdict level (median Cohen’s κ = 0.22), and system rankings did not survive a change of judge: Kendall’s τ with the human ranking ranged from 0.17 to 0.89 (median 0.67), so the preregistered hypothesis that rankings remain stable was not supported. Conclusions replicated by two low-cost judges from different providers matched the human conclusion more often than single-judge conclusions (+0.10 precision), but the gain came from filtering out one poorly aligned judge, and replication involving that judge left a conclusion for only 35% of comparisons. Judges that agreed with humans reached a stable conclusion with no more verdicts than human votes required; the others failed to converge on 26–70% of system pairs at the sample sizes tested (up to 100 verdicts per pair). Disagreements with the human vote co-occurred across judges far more often than independence predicts, although a single human vote per battle cannot separate shared judge bias from voter noise.

## 1 Introduction

LLM judges are now a routine way to compare language systems (Zheng et al., 2023). A typical use is a pairwise comparison: a judge model reads one request and two responses and says which response is better, and the verdicts are aggregated into a win rate or a ranking. When human labels are unavailable, the only check on such a conclusion is another judge. A common recommendation is to combine judges from different model families, on the expectation that their individual biases differ (Verga et al., 2024).

This paper asks two practical questions about that check. First, when a system-level conclusion obtained with one judge is reproduced by a judge from a different provider, is it more likely to match the conclusion that human votes would give? Second, how many verdicts does a judge need before its system-level conclusion is stable, compared with the number of human votes needed for the same comparison?

A widely repeated observation motivates optimism: judges can disagree with humans on many individual items while still ranking systems similarly (Gera et al., 2025; Gao et al., 2025). We preregistered this observation as a replication hypothesis (H1) and three further hypotheses on replication (H2), sample requirements (H3), and co-occurring disagreement (H4). The registered design had six judges; one was dropped after registration, as described in Section 3.6. We then re-judged 4,000 battles from a public Chatbot Arena release (Chiang et al., 2024; LMArena, 2025) with five judge models from three providers. Every battle has a human vote, so verdict-level and system-level quantities are computed on the same items.

The results are mixed, and we report them against the preregistered criteria.

• H1 was not supported. Verdict-level agreement with humans was low for every judge, and rank correlation with the human ranking depended strongly on the judge (τ from 0.17 to 0.89; median 0.67 against a preregistered threshold of 0.8).

• H2 was supported, with a narrow mechanism. Replicated conclusions were more precise than single-judge conclusions, but only because replication removed the wrong conclusions of one poorly aligned judge. Two well-aligned judges gained nothing from replicating each other, and replication with the poorly aligned judge reduced coverage from 100% to 35% of comparisons.

• H3 held for four of five judges. Where a judge converged, it needed no more verdicts than human votes. Three judges failed to converge on a quarter or more of the system pairs within the sizes tested.

• H4 was supported statistically, but its interpretation is limited because the reference label is a single vote from one user.

The study is small (nine systems, 33 system pairs, three low-cost judges in the cross-provider analyses) and its contribution is correspondingly modest: a preregistered measurement, on humanvoted battles, of what cross-provider replication buys and what it costs at the level of systemcomparison conclusions. We release battle identifiers, all verdicts, the preregistration and deviation records, and the analysis code as ancillary files with this preprint.

## 2 Related Work

Verdict-level versus system-level judge quality. Gera et al. (2025) evaluate 48 judge realizations by how well the system ranking they induce matches a human-based leaderboard, and show that verdictlevel and system-level quality can diverge. Gao et al. (2025) compare combinations of input sets, judges, comparison formats, and aggregation methods and report a similar divergence. Brocai and Becker (2026) show for text ranking that disagreement concentrates on pairs with small quality gaps, which limits its effect on rankings. Our H1 restates the common reading of this line of work as a threshold test and finds that it fails for the judges and data studied here; unlike the leaderboard-based evaluations, our human reference consists of votes on the same battles the judges see.

Judge bias and sample efficiency. Dorner et al. (2025) prove that when a judge is no more accurate than the systems it evaluates, bias correction cannot save more than half of the ground-truth labels, and show that judge bias can perturb rankings even when judge accuracy is high. Our H3 analysis is an empirical counterpart on arena data: it compares, pair by pair, the number of judge verdicts and human votes needed to reach the same directional conclusion.

Panels and correlated errors. Panels of diverse judges were proposed to reduce the bias of any single judge (Verga et al., 2024). Kohli (2026) report that errors of panel members are strongly correlated on NLI items and that the best single judge matches or beats the panel. We observe the same two patterns for pairwise system comparison: disagreements with the human vote co-occur across judges, and a five-judge majority does not beat the best single judge.

Self- and family-preference. Judges can favor their own generations (Panickssery et al., 2024), and recent work measures preference at the level of model families (Awuni et al., 2026). A recent analysis by the Arena team had twelve models judge 1,460 English Text Arena battles from 2026 and compared the verdicts with the human votes (Arena Team, 2026). It reports that most judges favor their own responses and those of sibling models, that judges agree with each other far more often than with the human voter, and that even on the battles where all twelve judges chose the same response the voter agreed only 60.6% of the time. That analysis is at the verdict level and includes battles in which the judge wrote one of the responses; it does not examine system rankings, replicated systemlevel conclusions, or sample requirements. Our provider-preference analysis is a prespecified secondary analysis that is consistent with these reports, and we do not present it as a new finding.

## 3 Study Design

## 3.1 Data and sample

We use a public Chatbot Arena release, arena-human-preference-140k (LMArena, 2025), pinned to a fixed revision, which contains 135,634 battles. A battle consists of one user request, the responses of two anonymous systems, and the user’s vote. We keep single-turn battles, the first vote of each conversation, votes that choose one of the two responses (ties and “both bad” are excluded), and battles whose request and two responses total at most 6,000 tokens; 65,636 battles remain (Appendix A). We do not filter by language. Restricting to English left only three system pairs with at least 100 battles, so the language restriction was dropped before any judge output was produced, and results are reported separately for English and non-English battles.

Systems were added in decreasing order of battle count until system pairs with at least 100 battles could supply 5,000 battles under a common per-pair cap. This rule selected nine systems and 33 system pairs with 111–183 battles each $( \mathsf { A p - }$ pendix B); the comparison graph is connected. The sample was split by pair into a development set (1,000 battles) used for pipeline checks and a confirmation set (4,000 battles) used for all results in this paper. Of the confirmation battles, 2,156 are English, 1,471 are in another identified language, and 373 have no identified language.

## 3.2 Judges

Five judge models from three providers were used: Claude Opus 5.5, Claude Sonnet 5.5, and Claude Haiku 4.5 (Anthropic); GPT-6 Luna (OpenAI); and Gemini 3.1 Flash-Lite (Google). The three low-cost models from different providers (Haiku, Luna, Flash-Lite) form the economy set used for the cross-provider analyses, so that provider differences are not confounded with tier differences. The two larger Anthropic models show the effect of tier within one provider. Exact model identifiers and request settings are given in Appendix C.

## 3.3 Judging protocol

All judges received the same prompt (Appendix C), which shows the request and the two responses without system names and asks for a brief justification followed by a final line with one of three verdicts: A, B, or tie. Each battle was judged twice with the response order swapped. If the two verdicts selected the same system, that system wins; if they selected different systems or either was a tie, the resolved verdict is a tie. We call a battle order-consistent when the two orders selected the same system or were both ties. A battle with an unparseable or refused response in either order is missing for that judge. For 500 confirmation battles, both orders were judged two additional times to measure run-to-run consistency. Each judge therefore produced 10,000 confirmation responses.

## 3.4 Measures

For pair-level quantities, scores are oriented to the alphabetically first system X of each pair: 1 if X wins, 0 if it loses, and 0.5 for a tie. For a judge or for the human votes, the win rate w of a pair is the mean score, and its direction is the sign of $w - 0 . 5$ A direction is significant if $| w - 0 . 5 | > 1 . 9 6 \mathrm { s } \epsilon$ e, where se is the standard error of the mean. We write dir<sub>H</sub> and $g _ { H } = | w _ { H } - 0 . 5 |$ for the human direction and gap.

Verdict level. Cohen’s κ (Cohen, 1960) between a judge’s resolved verdicts and the human votes over the categories {first system, second system, tie}, in the original left/right orientation of each battle.

System level. Bradley–Terry strengths (Bradley and Terry, 1952) fitted with an MM algorithm (Hunter, 2004), with ties counted as half a win for each side and 0.5 pseudo-wins added in each direction for every observed pair, and Kendall’s $\tau _ { b }$ (Kendall, 1938) and Spearman’s $\rho$ between the judge-based and human-based rankings.

Conclusion level. Let E be the set of pairs whose human direction is significant (23 of 33 pairs in the confirmation set). The single-judge precision $P _ { 1 } ( j )$ is the fraction of pairs in E with a significant direction for judge j whose direction equals dir<sub>H</sub>. The replicated precision $P _ { 2 } ( j , k )$ is the same fraction over pairs for which judges j and k are both significant in the same direction. Coverage is the fraction of pairs in E that yield a conclusion.

Sample requirement. For a source (human votes or a judge), a pair, and $n ~ \in ~ \{ 2 5 , 5 0 , 1 0 0 \}$ , we draw 1,000 samples of n battles with replacement and record the fraction whose direction equals the full-data dir . The requirement $n ^ { * }$ is the smallest n for which this fraction is at least 0.95; if no n reaches it, $n ^ { * }$ is censored. A size n is skipped for a source when it exceeds that source’s number of valid battles for the pair. The preregistered grid also contained $n = 2 0 0$ , which exceeds every pair’s confirmation size, and $n = 1 0 0$ was skipped for the five pairs with fewer than 100 confirmation battles.

Co-occurring disagreement. Over battles on which all three economy judges give a non-tie verdict, the observed rate O at which all three disagree with the human vote is compared with the product $E _ { \perp }$ of their individual disagreement rates.

## 3.5 Hypotheses

H1 The median κ over judges is below 0.6 and the median τ is above 0.8.

H2 $\Delta = \overline { { P _ { 2 } } } - \overline { { P _ { 1 } } } > 0$ , where the means are over the cross-provider pairs and the members of the economy set.

H3 For pairs with $g _ { H } \geq 0 . 1$ and a finite human $n ^ { * }$ a judge’s median ratio $n _ { \mathrm { j u d g e } } ^ { \ast } / n _ { \mathrm { h u m a n } } ^ { \ast }$ is at most 2, with censored values treated as infinite.

$$
{ \bf H } 4 D = O - E _ { \perp } > 0 .
$$

H2 and H4 are tested with one-sided bootstrap tests and Holm correction (Holm, 1979); H1 and H3 are decided by the stated thresholds. Each of the 1,000 bootstrap replicates resamples battles with replacement within pairs, jointly for all judges and the human votes; the set E is held fixed; p-values use the plus-one estimator; and precision terms with a zero denominator are excluded from the means and counted.

## 3.6 Preregistration and deviations

The prompt, parser, verdict-resolution rule, sample, hypotheses, and analysis code were hashed and frozen before any confirmation response was requested. The set E was fixed from human votes alone. Two changes were made after the freeze. First, the registered design had six judges, including an open-weight model served by a fourth organization as a fourth economy judge. That judge was dropped for a logistical reason (no additional hosting account was opened), and the decision was taken after the confirmation results of the three Anthropic judges had been seen. H1 is therefore evaluated over five judges and H2 and H4 over three economy judges. Second, after the freeze, the majority-vote rule was changed from the original requirement that all judges have valid verdicts to an abstention rule: ties and missing verdicts abstain, A and B votes are counted, and equal counts (including no votes) give a tie. Both changes were recorded on October 6, 2026, after the Anthropic confirmation results had been seen and before the full five-judge analysis. The majority vote does not enter any hypothesis criterion. No other definition changed. Analyses marked post hoc below were not preregistered.

## 4 Results

## 4.1 Response quality

All 50,000 confirmation responses were retrieved. Parse failures were at most four per judge. Opus and Sonnet refused 76 and 39 requests (0.76% and 0.39%); the other judges refused none. Resolved verdicts were available for $3 , 9 7 1 { - } 4 , 0 0 0$ of the 4,000 battles per judge.

## 4.2 Verdicts and rankings (H1)

Table 1 gives the verdict-level and system-level agreement of each judge with the human votes. The median κ is 0.217 (95% bootstrap interval 0.193– 0.240), below the 0.6 threshold. The median τ is 0.667 (0.556–0.722), below the 0.8 threshold, so H1 is not supported. Two judges reach $\tau = 0 . 8 8 9 .$ two reach 0.667, and one reaches 0.167. With nine systems, the difference between 0.889 and 0.667 corresponds to four more of the 36 system pairs being ordered differently from the human ranking.

The pattern is similar for English battles (τ from 0.111 to 0.944) and non-English battles (0.167 to 0.889; Appendix D). In a post hoc analysis that leaves out the least aligned judge, the median τ over the remaining four is 0.778, still below the threshold.

Three further observations qualify these numbers. First, the two low-cost judges Haiku and Flash-Lite agree with human votes more closely than the two larger Anthropic judges in κ and in both rank correlations. This says that their preferences are closer to those of Arena voters; it does not show that their verdicts are more correct. Second, the majority vote of the five judges $( \tau = 0 . 7 7 8 )$ does not beat the best single judges. Third, for every judge the observed τ lies inside the 95% range of a random-error model that keeps the judge’s tie rate and its agreement rate on non-tie verdicts but places disagreements at random. This does not show that the disagreements are random; it shows that with nine systems the ranking statistic alone cannot tell systematic disagreement from random disagreement. The provider-level analysis in Section 4.6 can.

## 4.3 Replication across providers (H2)

Table 2 reports precision and coverage on the 23 pairs in E. The mean replicated precision is 0.902 and the mean single-judge precision of the economy set is 0.804, giving $\Delta = 0 . 0 9 8$ (95% interval 0.052–0.194; Holm-adjusted $p = 0 . 0 0 2 )$ . H2 is supported. On all 33 pairs with a nonzero human direction (secondary analysis), $\Delta = 0 . 1 2 5 ( 0 . 0 2 8 -$ 0.172).

The table also shows where the gain comes from. Haiku and Flash-Lite each reach the human conclusion on 22 of 23 pairs, and the conclusions they share are right on the same 22 of 23: replication between these two judges adds nothing, and the one wrong conclusion is common to both. Luna reaches a significant conclusion on 14 pairs and is right on seven. Requiring agreement with a second judge removes most of Luna’s wrong conclusions (seven of eight replicated conclusions are right), but only eight of 23 comparisons then have a conclusion at all. Replication thus acted as a filter against a poorly aligned judge, at a large cost in coverage, and not as a general improvement in validity. A user cannot tell from replication alone which of the two situations applies: agreement between Haiku and Flash-Lite and agreement between Haiku and Luna look the same from the inside. The verdictlevel analogue has been reported by Arena Team (2026), where unanimous agreement among twelve judges still matched the human vote on only 60.6% of battles.

<table><tr><td>Judge</td><td>Provider</td><td>κ</td><td> $\tau _ { b }$ </td><td> $\rho$ </td><td>Order-consistent</td><td>Tie</td><td>Repeat</td></tr><tr><td>Opus 5.5</td><td>Anthropic</td><td>0.202</td><td>0.667</td><td>0.783</td><td>89.4%</td><td>11.6%</td><td>0.868</td></tr><tr><td>Sonnet 5.5</td><td>Anthropic</td><td>0.217</td><td>0.667</td><td>0.783</td><td>87.6%</td><td>13.3%</td><td>0.845</td></tr><tr><td>Haiku 4.5</td><td>Anthropic</td><td>0.266</td><td>0.889</td><td>0.967</td><td>83.0%</td><td>17.5%</td><td>0.942</td></tr><tr><td>GPT-6 Luna</td><td>OpenAI</td><td>0.052</td><td>0.167</td><td>0.217</td><td>78.5%</td><td>22.1%</td><td>0.807</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>Google</td><td>0.302</td><td>0.889</td><td>0.967</td><td>85.8%</td><td>14.9%</td><td>1.000</td></tr><tr><td>Majority of five</td><td></td><td>0.267</td><td>0.778</td><td>0.900</td><td></td><td></td><td></td></tr></table>

Table 1: Agreement with human votes on the confirmation set. κ: verdict-level agreement; $\tau _ { b } , \rho \colon$ correlation between judge-based and human-based Bradley–Terry rankings of nine systems. Order-consistent: battles where the two response orders selected the same system or were both ties. Tie: resolved ties. Repeat: share of repeat battles with three valid resolved verdicts (493–500 per judge) in which all three were identical.

![](images/eea8f52b7ea3259400ef084fc892f2612485ef47342ee276857cf5263e2db2f0.jpg)  
Figure 1: Verdict agreement and rank correlation with the human reference. Markers identify individual judges; the reference lines show the preregistered H1 thresholds.

<table><tr><td>Source of conclusion</td><td>Precision Coverage</td></tr><tr><td>Single judge</td><td>1.00</td></tr><tr><td>Haiku 4.5</td><td>22/23 (0.96)</td></tr><tr><td>GPT-6 Luna Gemini 3.1 Flash-Lite</td><td>7/14 (0.50) 0.61 22/23 (0.96) 1.00</td></tr><tr><td>Replicated across providers Haiku 4.5 +</td><td>7/8 (0.88) 0.35</td></tr><tr><td>GPT-6 Luna Haiku 4.5 +</td><td>22/23 (0.96) 1.00</td></tr><tr><td>Gemini 3.1 Flash-Lite GPT-6 Luna + Gemini 3.1 Flash-Lite</td><td>7/8 (0.88) 0.35</td></tr><tr><td></td><td></td></tr><tr><td>Not part of H2 Opus 5.5</td><td>15/17 (0.88) 0.74</td></tr><tr><td>Sonnet 5.5</td><td>17/19 (0.89) 0.83</td></tr><tr><td>Majority of five</td><td>20/21 (0.95) 0.91</td></tr></table>

Table 2: Precision and coverage of significant directional conclusions on the 23 system pairs whose human direction is significant.

## 4.4 Sample requirements (H3)

Table 3 gives the distribution of $n ^ { * }$ over the 23 pairs with $g _ { H } \ge 0 . 1$ ; human votes reach a stable direction on all of them. The median ratio of judge to human $n ^ { * }$ is 1 for Opus, Sonnet, Haiku, and Flash-Lite and infinite for Luna, so H3 holds for four of five judges. The grid is coarse, and a median ratio of 1 means only that judge and human typically fall in the same grid cell; it is not a statement about every pair.

The censored counts carry most of the information. Haiku and Flash-Lite reproduce the human direction with 25 verdicts on 22 of 23 pairs, while human votes need 50 or more on eight pairs: where these judges agree with the human direction, their verdicts are less noisy than individual votes. Sonnet, Opus, and Luna do not converge on 6, 8, and 16 pairs (26%, 35%, and 70%). For these pairs, the judge’s win rate is on the other side of 0.5 or too close to it for the sizes tested, so the available verdicts do not recover the human direction. This matches the theoretical account of Dorner et al. (2025), in which judge bias, not judge noise, limits what additional judge labels can deliver.

<table><tr><td>Source n*=25 50 100 Censored</td></tr><tr><td>Human votes 15 7 1</td></tr><tr><td>0 Haiku 4.5 22 0 0 1</td></tr><tr><td>Gemini 3.1 Flash-Lite 22 0 0 1</td></tr><tr><td>Sonnet 5.5 14 3 0 6</td></tr><tr><td>Opus 5.5 14 1 0 8</td></tr><tr><td>GPT-6 Luna 3 3 1 16</td></tr><tr><td>Majority of five 18 2 0 3</td></tr></table>

Table 3: Number of system pairs (of the 23 with human gap $g _ { H } \ge 0 . 1 )$ by the smallest sample size at which the source reproduces the human direction in at least 95% of resamples. Censored: not reached at any size tested.

## 4.5 Co-occurring disagreement (H4)

On the 2,351 confirmation battles where all three economy judges give a non-tie verdict, all three disagree with the human vote on 15.8% of battles, against 3.5% expected if their disagreements were independent $( D = 0 . 1 2 2$ , 95% interval 0.111– 0.133; Holm-adjusted $p \ = \ 0 . 0 0 2 )$ . H4 is supported.

This result needs a careful reading. The reference is one vote from one user. If that user preferred the response that most readers would consider worse, every competent judge “disagrees” at once, and the disagreements co-occur without any shared judge bias. Our data cannot separate this from shared bias. Two observations are consistent with a substantial contribution of voter noise: at the level of system pairs, where single-vote noise averages out, the economy judges share only one wrong conclusion among the 23 pairs in E (post hoc), and the co-occurrence rate is lower in pairs with larger human gaps (post hoc). We therefore describe H4 as co-occurring disagreement with the human vote and not as evidence of common judge error.

## 4.6 Provider preference (prespecified secondary analysis)

Because the nine selected systems are dominated by three providers, excluding battles that involve a judge’s own provider removes 31–63% of the sample, so we measure preference directly instead. Table 4 shows, for each judge and system provider, how much more often the judge favors that provider’s systems than human voters do. Opus, Sonnet, and Flash-Lite favor their own provider’s systems; Haiku disfavors them; Luna shows no own-provider preference but strongly favors Anthropic systems. All judges rate the Mistral and Alibaba systems lower than human voters do. The preferences are thus systematic and judgespecific, and they are not explained by provider identity alone. For example, human votes rank Claude Opus 4 seventh of nine systems, while the Opus and Sonnet judges rank it fourth. These measurements compare judges with Arena voters and say nothing about which side is right; they are consistent with recent reports of self- and familypreference (Panickssery et al., 2024; Awuni et al., 2026; Arena Team, 2026).

![](images/b22d7dd43e98975fe666610c76467e7acd7f39f0d0a08fe83745390c4c5cedd8.jpg)  
Figure 2: Within-pair bootstrap direction agreement, averaged over eligible pairs in each human-gap group. Pair membership can change with sample size because sizes exceeding a source’s valid battle count are skipped. The dotted reference is 0.95.

## 5 Discussion

What replication with a second provider can and cannot show. In this study, agreement between two judges from different providers was informative only in one direction. When a poorly aligned judge was involved, disagreement with a second judge flagged most of its wrong conclusions. But agreement between two judges did not certify a conclusion: the two best-aligned judges shared their one wrong conclusion, and the pair-level precision of replicated conclusions was no higher than that of the better judge alone. Replication is therefore better understood as a screen for an unreliable judge than as a substitute for human validation, and its cost in lost conclusions can be large.

<table><tr><td>Judge</td><td>Anthropic</td><td>Google</td><td>OpenAI</td><td>Mistral</td><td>Alibaba</td><td>Own minus others (95% CI)</td></tr><tr><td>Opus 5.5</td><td>+0.101</td><td>+0.081</td><td>-0.019</td><td>-0.159</td><td>-0.190</td><td>+0.173 [0.141, 0.211]</td></tr><tr><td>Sonnet 5.5</td><td>+0.072</td><td>+0.091</td><td>+0.002</td><td>-0.154</td><td>-0.169</td><td>+0.129 [0.098, 0.163]</td></tr><tr><td>Haiku 4.5</td><td>-0.104</td><td>+0.126</td><td>+0.099</td><td>-0.066</td><td>-0.032</td><td>-0.136 [-0.167, -0.105]</td></tr><tr><td>GPT-6 Luna</td><td>+0.245</td><td>-0.114</td><td>-0.074</td><td>-0.125</td><td>-0.145</td><td>-0.039 [-0.082, 0.003]</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>-0.110</td><td>+0.155</td><td>+0.125</td><td>-0.105</td><td>-0.062</td><td>+0.193 [0.161, 0.224]</td></tr></table>

Table 4: Mean difference between the judge’s score and the human score for systems of each provider, over battles in which exactly one of the two systems is from that provider (870–2,163 battles per cell). Positive values mean the judge favors that provider’s systems more than human voters do. Bold: the judge’s own provider. Last column: own-provider cell minus the mean of the other cells, with 95% bootstrap interval.

Judge choice matters more than sample size. For well-aligned judges, 25 verdicts per system pair were enough on almost all pairs with a human gap of at least 0.1. For the other judges, a quarter or more of the pairs never converged. Within the sizes we tested, more verdicts from the same judge did not resolve these pairs; a small human-labeled set that reveals which judge is aligned for the task at hand would address the cause directly.

Stronger models were not better aligned with voters. The two larger Anthropic judges agreed less with Arena votes than the low-cost judges did. One reading is that larger models apply criteria that differ from those of casual voters, for example by weighting correctness over presentation. Our data do not test this reading.

## 6 Limitations

The study has nine systems and 33 system pairs, and rank correlations over nine systems are coarse. Conclusion-level results rest on 23 pairs, and the H2 effect depends on a single poorly aligned judge. The cross-provider analyses use one low-cost judge per provider; we did not test replication between larger models from different providers. The registered design included a sixth judge, an open-weight economy model, that was dropped after partial results had been seen; although the reason was logistical, this is a deviation from the preregistration.

All five judges are proprietary models that may be retired, which limits exact reproduction of the verdicts; we release the verdicts themselves.

The human reference is a single crowd vote per battle from 2025. Such votes are noisy and reflect voter preference, not correctness, and the judged systems are older than most of the judge models. Some battle texts may have been seen by the judges during training. The judging prompt is in English while 37% of confirmation battles are in another identified language. Ties and “both bad” votes were excluded, so the results apply to battles where the voter expressed a preference. We tested one prompt and one verdict-resolution rule; treating order-inconsistent verdicts as ties is one of several reasonable choices. Reasoning settings were not uniform across judges, because providers expose different controls (Table 6): GPT-6 Luna ran with reasoning turned off, while Opus and Sonnet ran with low-effort reasoning. We cannot tell whether Luna’s low agreement with human votes reflects the model or this setting, and the H2 effect depends on that judge. The prediction-powered baseline listed in the preregistered plan was not implemented.

## Ethics and Reproducibility

The data are a public release of user conversations collected between April and July 2025 with preexisting votes; no new human annotation was collected. The dataset documentation does not describe removal of personal information. We did not apply further filtering; the texts were processed only through the providers’ batch interfaces, and we do not redistribute request or response text. The release contains battle identifiers, all judge verdicts with token usage, the frozen analysis code, the preregistration, and the deviation record, from which every number in this paper can be regenerated. The code is released under the MIT License and the verdicts and derived tables under CC BY 4.0; the underlying requests and responses remain under the terms of the original release. Judging cost about USD 157 at the fixed exchange rate used in the cost ledger.

Use of AI assistants. The judging and analysis pipeline was implemented with an LLM-based coding agent, and LLM assistants were used in planning the analysis and in drafting this manuscript. The authors reviewed the code through its test suite and an independent reproduction run, checked the reported numbers against the released tables, and take responsibility for the content.

## References

Arena Team. 2026. Self-preference of AI judges. Arena blog, published 30 September 2026.

David Ababio Awuni, Luke E. K. Achenie, Benjamin Tei Partey, Elvis Gyasi Owusu, and Nii-Nai Derrick Sowah. 2026. Who judges matters: Measuring family-conditioned preference in LLMas-judge panels. Preprint, arXiv:2609.17857.

Ralph Allan Bradley and Milton E. Terry. 1952. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324– 345.

Bruno Brocai and Maria Becker. 2026. Pair difficulty matters: Rethinking pairwise LLM-as-a-judge evaluation and consistency. Preprint, arXiv:2609.37577.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. 2024. Chatbot Arena: An open platform for evaluating LLMs by human preference. Preprint, arXiv:2403.04132.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Florian E. Dorner, Vivian Y. Nastl, and Moritz Hardt. 2025. Limits to scalable evaluation at the frontier: LLM as judge won’t beat twice the data. In International Conference on Learning Representations.

Mingqi Gao, Yixin Liu, Xinyu Hu, Xiaojun Wan, Jonathan Bragg, and Arman Cohan. 2025. Reevaluating automatic LLM system ranking for alignment with human preference. In Findings of the Associationfor Computational Linguistics: NAACL 2025, pages 4605–4629, Albuquerque, New Mexico. Association for Computational Linguistics.

Ariel Gera, Odellia Boni, Yotam Perlitz, Roy Bar-Haim, Lilach Eden, and Asaf Yehudai. 2025. JuStRank: Benchmarking LLM judges for system ranking. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 682–712, Vienna, Austria. Association for Computational Linguistics.

Sture Holm. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal ofStatistics, 6(2):65–70.

David R. Hunter. 2004. MM algorithms for generalized Bradley-Terry models. The Annals of Statistics, 32(1):384–406.

Maurice G. Kendall. 1938. A new measure of rank correlation. Biometrika, 30(1/2):81–93.

Guneet Kohli. 2026. Nine judges, two effective votes: Correlated errors undermine LLM evaluation panels. Preprint, arXiv:2605.29800.

LMArena. 2025. arena-human-preference-140k. Hugging Face dataset, revision 6322995ab34d7c2693e3f47dd13fa5caa0789a74.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM evaluators recognize and favor their own generations. Preprint, arXiv:2404.13076.

Pat Verga, Sebastian Hofstatter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. 2024. Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. Preprint, arXiv:2404.18796.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and Chatbot Arena. Preprint, arXiv:2306.05685.

## A Filtering

<table><tr><td>Stage</td><td>Battles remaining</td></tr><tr><td>Raw release Duplicates removed</td><td>135,634 135,619</td></tr><tr><td>Two different systems Non-empty responses</td><td>135,609 135,472</td></tr><tr><td>Single turn</td><td>116,307</td></tr><tr><td>First vote per conversation</td><td>92,488</td></tr><tr><td>Vote selects one response</td><td>67,617</td></tr><tr><td>At most 6,000 tokens</td><td>65,636</td></tr></table>

Table 5: Filtering stages. Token counts use the cl100k\_base tokenizer.

After filtering, 146 system pairs have at least 100 battles. Identical request strings are rare in the

sample (39 of 5,000 battles belong to a group of repeated requests) and were kept.

## B Systems

The nine systems, in the order of the human-based Bradley–Terry ranking on the confirmation set:

1. gemini-2.5-pro   
2. o3-2025-04-16   
3. chatgpt-4o-latest-20250326   
4. gemini-2.5-flash   
5. qwen3-235b-a22b-no-thinking   
6. mistral-medium-2505   
7. claude-opus-4-20250514   
8. claude-sonnet-4-20250514   
9. claude-3-7-sonnet-20250219-   
thinking-32k

## C Judging protocol

All requests were sent through the providers’ batch interfaces. The target for visible output was 1,024 tokens; for Opus and Flash-Lite the total output limit was set to 5,120 tokens to leave room for reasoning tokens, which does not guarantee a fixed amount of visible output. An initial limit of 300 tokens truncated some responses in a partial development run; the limit was raised and the development run repeated before the freeze. Table 6 reports the settings used. Analyses used Python 3.11 with NumPy 2.4.6, SciPy 1.17.1, and pandas 3.0.6; the Bradley–Terry fit, bootstrap, and resampling procedures were implemented directly, and all package versions are pinned in the released lock file.

System prompt. You are an impartial evaluator comparing two AI assistant responses to the same user request.

User prompt. The request and the two responses are inserted under the headings [User request], [Response A], and [Response B], followed by: Which response better serves the user’s request? Give a briefjustification in at most three sentences. Then end with exactly one final line in one of these forms: VERDICT: A /VERDICT: B /VERDICT: TIE.

After the analysis, all 60,000 development and confirmation responses were downloaded again from the providers and matched to their requests by the server-side request identifier; no mismatch was found (post hoc integrity check).

D English and non-English battles
<table><tr><td>Judge</td><td>κ (EN)</td><td>τb (EN)</td><td>κ (non-EN)</td><td>τb (non-EN)</td></tr><tr><td>Opus 5.5</td><td>0.180</td><td>0.611</td><td>0.221</td><td>0.556</td></tr><tr><td>Sonnet 5.5</td><td>0.212</td><td>0.667</td><td>0.219</td><td>0.667</td></tr><tr><td>Haiku 4.5</td><td>0.264</td><td>0.944</td><td>0.284</td><td>0.889</td></tr><tr><td>GPT-6 Luna</td><td>0.047</td><td>0.111</td><td>0.045</td><td>0.167</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>0.292</td><td>0.944</td><td>0.325</td><td>0.833</td></tr><tr><td>Majority of five</td><td>0.258</td><td>0.722</td><td>0.281</td><td>0.778</td></tr></table>

Table 7: H1 measures on English (2,156) and non-English (1,471) confirmation battles; 373 battles without an identified language are excluded from this table only.

<table><tr><td>Judge</td><td>Model ID</td><td>Temperature</td><td>Reasoning</td><td>Cap</td><td>Batch</td></tr><tr><td>Opus 5.5</td><td>claude-opus-5-5</td><td>Not sent (unsupported)</td><td>adaptive; effort=low</td><td>5120</td><td>Yes</td></tr><tr><td>Sonnet 5.5</td><td>claude-sonnet-5-5</td><td>Not sent (unsupported)</td><td>between_tools; effort=low; no tools</td><td>1024</td><td>Yes</td></tr><tr><td>Haiku 4.5 GPT-6 Luna</td><td>claude-haiku-4-5-20251001</td><td>0</td><td>disabled</td><td>1024</td><td>Yes</td></tr><tr><td>Gemini 3.1</td><td>gpt-6-luna</td><td>Not sent (unsupported)</td><td>effort=none</td><td>1024</td><td>Yes</td></tr><tr><td>Flash-Lite</td><td>gemini-3.1-flash-lite</td><td>0</td><td>thinkingLevel=minimal</td><td>5120</td><td>Yes</td></tr></table>

Table 6: Execution settings for the five judges. Cap is the API output-token limit, including reasoning where applicable. “Not sent” denotes an omitted temperature parameter, not a claimed provider default. All requests used batch interfaces.