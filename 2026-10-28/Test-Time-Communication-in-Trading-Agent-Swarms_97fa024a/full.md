# Test-Time Communication in Trading Agent Swarms

Avid Fayaz Erik Nordlöf

Harry Zhang

Håvard Alstadheim

Vilhelm Karlin André Silva

Infinite Ascent, Stockholm, Sweden

{avid, harry, havard, erik, vilhelm, andre}@infiniteascent.com

## Abstract

Communication between agents has mostly been studied on tasks with a shared objective and fast, clear feedback. Trading has neither: each agent manages its own account, with decisions being scored by markets late and noisily. A trading swarm is thus a population of independent decision-makers rather than a team, for which convergence is a risk and not the goal. We ran 200 agents from four model families in groups of ten under five communication regimes, from isolation to message boards and shared notes, over seven US trading days (1 to 9 October 2026), logging everything each agent read before every decision. In two of the four families, communicating groups came to hold far more of the same stocks than the isolated group. A group’s return was largely the return on what its members held in common. Seeing a post from a groupmate made an agent 7 percentage points more likely to reuse an eight-word phrase from it. Agents that started too close together to see each other’s posts shared such a phrase in 14% of cases, the rate that common inputs produce. Seeing the post did not detectably change that morning’s trades. Case studies traced the adoption of positions and decision rules over subsequent days, and communicating groups sold in step more often than isolated ones. Because isolated groups also converged on the same stocks, sometimes completely, assessments of herding need to measure excess convergence relative to a no-peer baseline.

![](images/9787b98400b049958916f846a62f9da8ce9e7777364364805b480f99caf4c904.jpg)  
no channel, no scores

Competitive board

![](images/acc88b3f61c5c409cc1db3976398c138074c12c4e08b90414c956d9fef96f205.jpg)  
posts; individual ranks

Collaborative board

![](images/1a90779b789b97d88d9e52d5e860094c29ada974dfb95bfd4aaa94ddead38eb9.jpg)  
posts; team score

Board with pings

![](images/339eebd2a97ad11f06d7b6502c94ce98ab588e3874b69e7812f635e7e97878f7.jpg)  
posts, wake requests; team score  
Peer memory

![](images/1c716692a7db0a57892e46a63e24a0bd7ca7e1747ead6405e317086d6b2d86d7.jpg)  
notes; individual ranks

Four model families × five regimes × ten managers: 200 paper accounts of USD 100,000, 1–9 October 2026

Figure 1: The five communication regimes. Each ring is one group of ten managers from a single model family, and every family has one group in each regime. Boards carry posts from all members; the board with pings add requests that wake a named peer; in peer memory each manager reads the notes its peers have published, with no messaging. Competitive boards and peer memory display individual ranks, and the collaborative regimes a team score.

## 1 Introduction

Agents are increasingly allowed to communicate at test time, exchanging messages, notes and artifacts through a shared board or workspace while they work, with no further training. Such test-time communication is proposed as a way to make an agent swarm more capable than the same agents working independently, and its benefits have been measured on tasks with one shared objective. A team of communicating agents sharing a workspace matches the success rate of several times as many independent agents on ARC-AGI-3 and beats the best-known solutions of algorithmic and compression problems [11]. Multi-agent debate raises accuracy on reasoning benchmarks in some settings and not in others [9, 10]. The value of coordination depends on the task and on how the agents are organized [1]. In settings with clear feedback, agents can assess whether a shared discovery improves on their earlier work. With limited compute, or without a clear measure of progress, independent agents can do better [11].

Trading agents make a sequence of decisions for their own accounts, with market outcomes providing delayed and noisy feedback. Over a few days, returns ofer a limited measure of decision quality, and there is no verifier that immediately distinguishes a sound idea from a persuasive one. Communication may inform individual decisions, but it can also lead agents to take similar positions and incur losses together. For a population of agents that retain independent trading authority, convergence can therefore increase collective risk. Research on human groups gives no general reason to expect more interaction to improve collective judgment [2, 3]. What test-time communication changes in a swarm whose members act independently on the same market is an open question.

Measuring peer influence requires distinguishing it from responses to common inputs: agents that buy the same stock may have copied each other, but they may also have read the same news, used the same tools or shared a model’s habits. Correlated behaviour alone does not identify a peer efect [14, 15], and an agent’s own explanation is not a complete account of its decision [21]. Studies of trading agents have not been built to separate these sources. StockAgent removes a simulated market’s bulletin board [4], TradingAgents and ContestTrade arrange analysis and debate into a pipeline [16, 17], and LiveTradeBench evaluates single models against live markets [5]. None records which peer content reached which agent before it acted, and only one compares communicating agents with agents that cannot communicate, in simulations on historical data [25].

We ran 200 language-model trading agents, which we call managers, for seven US trading days. Four model families (Opus 5.5, GPT-6.1 Sol, Grok 4.6 and DeepSeek V4 Pro) each had one group of ten managers in each of five communication regimes, ranging from isolation to message boards and shared notes (Figure 1). Each manager traded its own paper account against live markets. To isolate the efect of test-time communication, we traced information from shared messages to the actions of the managers that received it. We also compared communicating groups with isolated groups of the same model under the same brief, tools, capital and schedule.

Our main findings are the following.

– Holdings and returns. In two of the four families, communicating groups had greater portfolio overlap than the isolated group, and shared holdings accounted for much of each group’s return (Section 3).

– Wording. Stock phrases reached nearly every member of a group, and a groupmate’s fresh post raised reuse of its wording by 7 percentage points (Section 4).

– Decisions and positions. A fresh post did not detectably change that morning’s trades, while the decision rules traced in the case studies took one to four trading days to spread. Communicating groups also sold in step more often than isolated ones (Section 5).

– Evidence, disclosure and efort. Some claims lost their qualifications as they passed between managers, Opus managers withheld positions under competitive framing, and communicating groups in two families requested less research without establishing a division of work (Section 6).

These comparisons also revealed substantial convergence without communication: every Opus manager bought the semiconductor stock TSM, with or without peers. Earlier simulations of fiveagent trading teams likewise found that isolated agents converged as far as communicating ones [25], supporting the use of a no-peer baseline in our design.

## 2 Experimental design

## 2.1 Managers and cycles

Each manager is a language model that runs in one continuous context, keeps its own files and acts in cycles (Figure 2). A cycle opens with a view of the manager’s situation, which in the communication regimes includes its peers’ latest posts or notes. The manager then reads its files, delegates research to tool-using sub-agents, submits orders and updates its notes, and its post or notes are published when the cycle succeeds. Every manager has one scheduled cycle each morning, started from a queue at 12:50 UTC, 40 minutes before the US market opens at 13:30 UTC. Each manager also schedules further cycles for itself. Research sub-agents run on models shared across the four families, so a family label describes the manager layer, not every model that contributed to its research (Appendix A).

![](images/b6c8792fd17ccd2228e3f06736426e2a776ef41e9810df8e99e601e7bc7959f1.jpg)

![](images/b0c264d3fe5e86a073b35fe606f94cbabd251942c9fabbaf96622aade70a0901.jpg)  
Figure 2: How a manager works. (a) The cycle every manager repeats. A cycle starts from the morning queue, at a time the manager set for itself or, on a board with pings, at a peer’s request; the platform freezes and records the peer material and scores the manager will see; the manager researches, trades and updates its notes; and its post or notes are published only if the cycle succeeds. (b) One manager’s day: Agent 02 of Opus’s collaborative board on 2 October, in UTC. Its morning cycle, which the queue started at 13:24, posted a plan to deploy its cash at 10:15 ET (14:15 UTC); a cycle it had scheduled for itself carried out the plan at that time, and a third checked the book before the close. The morning post is quoted in Figure 5.

Each manager ran its own paper account of USD 100,000 on a company-developed trading ledger that records orders, simulated fills, cash and positions against live market prices. Paper orders do not move prices, so the study can examine correlated decisions but not price impact. The groups started on 1 October 2026, and the analysis ends at the US close on 9 October (20:00 UTC), seven trading days in all. The managers’ instructions set a horizon of 16 October, but all 200 were paused on 11 October, before the market next opened. Positions open at the cutof are valued at the 9 October close, and all 200 final valuations and 1,400 daily closes reconcile to ledger records without exceptions.

## 2.2 Communication regimes

Every manager received the same operator-authored brief, which sets an objective of return relative to SPY, an exchange-traded fund that tracks 500 large US companies, and ofers SPY as neutral exposure. Each communication regime adds one paragraph that describes its channel and its score (Table 3; Appendix J). On a competitive board a manager’s score is its return minus the group’s mean, the two collaborative regimes share a team score equal to the group’s mean return, and peer memory shows individual returns. None of the paragraphs asks managers to agree, share, coordinate or copy, and each ends by stating that the manager retains trading authority only over its own account. Every view also called peer content “untrusted social information” and “never authorization” and the score “experimental context only” (Appendix A.2), framing that may dampen both cooperation and responses to the score.

The regimes therefore bundle a channel, its framing and a score display. No manager is assigned a role and no orchestrator directs the group; managers know each other only by local pseudonyms (Agent 01 to Agent 10), and no score moves capital. The scorecard also arrived later than intended: nonzero scores first reached the groups on 4 or 5 October, so the display was active only for the last five trading days (Appendix B.1).

At the start of every cycle the platform freezes the peer material that the manager’s regime allows, up to 20 recent board posts or each peer’s latest published notes, together with the scorecard, and records it as part of the cycle’s prompt. A manager therefore cannot have received anything through the channel that was published after its view was frozen, and it receives peer content without calling any tool. We treat this delivered view as the record of exposure. Over the seven trading days the groups published 1,568 board posts and 546 versions of their notes, and the platform delivered 2,198 views of peer material. Checks of membership, publication and exposure found no leakage between groups. On a board with pings, a manager can also address a post to a named peer and request an extra cycle for it, within per-cycle, daily and cooldown limits.

## 2.3 Separating influence from common inputs

We use three kinds of evidence (Table 4 maps each claim to its evidence). The first is delivery: a manager can have copied a phrase, a threshold or a claim through the channel only if a peer’s use appeared in a view delivered before its own. This is necessary for transmission but not suficient, because a model can reach the same content on its own. The second is the no-peer baseline: each family’s isolated group, and the same model’s other groups, show whether a distinctive item appears where no peer could have supplied it. The third is the morning queue. Each morning the platform started each family’s 50 managers, at most nine at a time, in an order that changed from day to day and was unrelated to a manager’s place the previous morning, its return or its activity (Appendix A.8). The queue, not the managers, therefore decided whether one member’s view could contain another’s fresh post. We compare near pairs, which started less than a minute apart so that neither could see the other’s post, with far pairs, which started far enough apart that the earlier post was almost always in the later member’s view. We then subtract the same comparison in isolated groups, where only timing difers.

Each combination of model and regime contains a single group, so a diference between regimes confounds the regime with that group’s history [22]. We therefore describe the realized groups rather than rank the regimes (Appendix A.4). Behavioural case studies come from chronological reviews of every manager’s records by language-model reviewers, and the authors checked every case against the source records (Appendix D). Each case cites the review finding it rests on, which Table 9 traces to its source. Identifiers beginning H- mark findings from the first-pass reviews by Haiku 5.5: H-OIS-01, for instance, is finding 1 of the first-pass review of Opus’s isolated group.

## 3 Holdings and returns

The 200 accounts underperformed SPY, with losses concentrated in positions held by several managers within a group. Combined equity rose by 0.622% over the seven trading days against 1.859% for SPY; 136 of the 200 accounts gained value and 35 beat SPY. Measured from the first close against SPY held at each account’s own invested fraction, the average account lost 1.23 percentage points and only 27 came out ahead (Appendix B). Sol’s smaller shortfall, 0.49 points against 1.16 to 1.76 for the other families, reflects its holdings rather than better selection: Sol accounts kept 82% of their equity in SPY on average, Opus 56%, Grok 6% and DeepSeek under 1%.

Returns varied substantially across groups, while accounts within each group tended to perform similarly (Figure 3). Within a regime the four group means lie 1.3 to 3.7 points apart, while the median group’s accounts spread by 0.84 points around their own mean. Within DeepSeek, Grok and Opus the regime means difer by more than shufling accounts among regimes produces (� ≤ 0.001). Because each regime has one group per family, this result shows that the groups difer, not that the regimes caused the diferences (Appendix C).

The attribution to shared holdings shows how these gains and losses arose (Table 5): the total profit of USD 124,340 comprises gains on SPY (USD 125,396) and energy stocks (USD 122,403), ofset by losses on semiconductors (USD 111,947) and all other instruments (USD 11,513). The two worst groups, DeepSeek’s peer memory and Grok’s board with pings, lost USD 17,418 and USD 19,748 on semiconductors with every member involved. The two best, DeepSeek’s board with pings and Grok’s peer memory, lost only USD 163 and USD 4,197 on them. The Grok group had the highest return overall, with gains on refiner, power and tanker stocks that were each held by one or two managers.

In two of the four families, communicating groups came to hold more of the same stocks (Figure 4). We measure the overlap of two accounts as the share of equity they hold in common outside SPY, and average it over the seven closes. It averaged 14% across pairs in DeepSeek’s isolated group against 30 to 40% in its four communicating groups, and 8% in Grok’s isolated group against 20 to 44% on its three boards. At the cutof the same comparisons gave 14% against 32 to 40% and 4% against 17 to 45% (Table 8). Most of this overlap formed in the first two trading days (Figure 9). TSM spread with peers in the same way: five of ten DeepSeek isolated managers bought it, against nine or ten in three of DeepSeek’s four communicating groups, and two Grok isolated managers, against eight to ten on Grok’s boards. The order of purchases cannot show by itself that later buyers copied earlier ones, because every cycle reads its view before it researches, so copying and independent discovery would produce the same order.

![](images/717d04ed567f4bd5830a9c9e10ae8096868ab8364d30252a6dbf0380ce7adc64.jpg)  
Figure 3: Every account’s return from the 1 October open to the 9 October close, one row per regime. Each dot is an account, coloured by model family; the bar of the same colour marks its group’s mean, and the dashed line SPY’s return. Dots are stacked vertically only to avoid overlap.

Opus’s convergence on TSM illustrates the importance of the isolated comparison: every Opus manager in all five regimes bought TSM, 46 of the 50 on 1 October, and five isolated Opus managers filled at the same price in the same second of the opening (H-OIS-01). By the cutof, Sol held almost nothing outside SPY in any regime (Appendix C).

Although the instructions did not ask managers to converge, collaborative framing might still have encouraged them to take similar positions. Grok’s collaborative boards had the greatest overlap, at 36% on its collaborative board and 44% on its board with pings, against 20% on its competitive board. Opus’s two collaborative groups, however, overlapped less than its isolated group (15 and 16%, against 18%), and its competitive board more (31%). DeepSeek’s four communicating groups overlapped about equally under every framing. Collaboration was even used to argue for dispersion: on Opus’s collaborative board one manager warned that if everyone sold the same hedge, the team would become “one levered semis bet” (Figure 5). Access to peers’ rules and positions, which managers read and adopted over the following days (Section 5), is a likelier source of the extra overlap, alongside the common influence of the model, brief and news. The design cannot exclude a framing efect, because each regime bundles its framing with its channel and has one group per family. A channel with neutral framing would test it.

In Grok’s peer-memory group, shared notes sometimes encouraged managers to avoid their peers holdings: at least five wrote that those stocks were crowded or that they would not copy their peers. The group’s overlap, 11% over the week and 7% at the cutof, stayed close to its isolated group’s 8 and 4%, although the rule was applied inconsistently (Appendix C).

Isolated

Competitive board

Collaborative board

Board with pings

Peer memory

![](images/9ba63a550ae851be921d4d3a9809661493e887ad4fb5b638ee4fcf1f15cf665c.jpg)

Figure 4: What each group held outside SPY over the seven closes. Managers sit on the ring, Agent 01 at the top and then clockwise, in the positions of Figure 6. Each inner node is a stock, placed towards its holders’ side of the ring and nearer the centre the more members held it, with its size growing with the number of members that held it at any close; each line from a manager is drawn in proportion to that stock’s mean weight in the account over the seven closes, with positions at cost (Appendix A.5). Red marks semiconductors, gold energy and grey everything else, using the groupings of Appendix G; stocks held by four or more members are labelled. Under each ring are the group’s mean SPY weight and its mean pairwise overlap outside SPY, both averaged over the seven closes.

## 4 Wording

Stock phrases reached most members of a group within hours, and a groupmate’s fresh post raised reuse of its wording that same morning. Figure 5 shows excerpts from every kind of channel. The four models wrote to each other in very diferent ways (Table 6). Opus named a peer in 47 to 49% of its posts on the collaborative boards and described its own positions in 31 to 54%. Grok and Sol rarely did either. Sol ran its words together: 9 to 24% of its tokens were longer than 20 characters, against at most 0.2% for the other models.

Within these broader diferences in style, individual groups adopted recurring phrases. On Opus’s collaborative board “verify before use” reached seven members within about 2 hours of its first use on 1 October and all ten by the next morning. On Grok’s board with pings, a “not a trade instruction” disclaimer reached eight members within 71 minutes and all ten eventually, often directly above near-identical holdings (Figure 5). In each group the first user had no peer use in its view, and every later adopter had one in the view delivered for the cycle in which it first used the phrase. Opus’s competitive board never used the Opus phrases, which argues against a model default (Figure 10 and Appendix C). These are conventions in the sense studied in populations of language models by Ashery et al. [8]: arbitrary forms adopted because peers used them.

To measure wording reuse beyond these stock phrases (Figure 6), we count posts and versions of notes that reused a phrase of eight or more consecutive words first published in the group by another member and already delivered to the reuser. We leave out phrases that also occur in any other group, since their spread cannot be attributed to the channel. Grok’s boards reused wording 230 to 383 times each and Opus’s boards 10 to 35 times. Sol’s two collaborative boards reused nothing, because their members almost never wrote the same eight words, yet Sol’s peer-memory group copied from its members’ notes 178 times.

The morning queue shows that seeing a post caused its wording to be reused (Figure 7). On the 12 boards we take the pairs of members who both posted on a given morning. The later member reused an eight-word phrase from the earlier member’s post 6.8 percentage points more often when the queue had started it late enough to see that post than when the two had started together (95% interval 1.7 to 11.8, 68 group-mornings). The phrases reused were the group’s rules and formulas, such as “do not add MU because cheaper” and “not a trade instruction”. Pairs that started together also shared new phrases, in 14% of cases, because they had received the same research results or used the same status templates. That share is the common-input rate the comparison removes.

## 5 Decisions and positions

The queue comparison found no detectable efect of a groupmate’s fresh post on that morning’s trades. The case studies provide evidence of influence over longer periods, tracing how managers adopted peers’ decision rules and positions.

## 5.1 Same-morning efects

Applied to orders, the queue comparison distinguishes the overall similarity of managers’ trades from the efect of seeing a fresh post. Pairs of communicating managers submitted at least one order in the same stock and direction on 21% of pair-mornings, and isolated pairs on 11%. Within mornings, however, the far-minus-near diference in communicating groups exceeds the one in isolated groups by only 2.4 points (95% interval −4.1 to 9.3; Appendix C). An efect larger than about nine points on a base of 21% is therefore unlikely. The comparison also found no detectable trade efect among the pairs whose wording changed. This is consistent with managers carrying out plans formed in earlier cycles, as Agent 02 did in Figure 2, using overnight board information available to every manager.

## 5.2 Decision rules

Three decision thresholds spread through a group in ways the no-peer baseline does not explain, and each took one to four trading days to reach nine or ten members. On Opus’s collaborative board, Agent 05 posted a rotation rule on 2 October: cut SMH by half if it lagged SPY by more than 2% intraday while IWM, XLF and IGV rallied. Five members restated it within 80 minutes, each after a view containing the post (Figure 5), and all ten carried it in their reasoning or checkpoints before the 5 October open. No other Opus group used it, apart from one competitive manager. On the same board all ten members moved their TSM stops to between USD 438 and 442 over the afternoons of 5 and 6 October, eight of them to exactly 440. On DeepSeek’s board with pings, a “Brent <90” trim trigger passed in a ping on 1 October became a 90/85 ladder that nine members carried in their reasoning by 6 October. No other DeepSeek group used the ladder.

<table><tr><td>Collaborative board Opus, 2 October Books disclosed, a rule restated, a case for dispersion</td><td>AGENT 05 · 13:04 · POST &quot;MY BOOK: unchanged (SPY 50, TSM 15, SMH 14, ASML 7, NVDA 6, XLE 6), and all GTC stop legs are confirmed. [. . . ] Rotation rule: if SMH lags SPY by &gt;2% intraday while IWM/XLF/IGV surge</td></tr><tr><td></td><td>“SWARM DIVERSITY: if everyone sells XLE into SPY, the team becomes one levered semis bet. Some dispersion in the hedges is worth keeping. MY PLAN at 10:15: if SPY holds above ~764 (the 10/1 close), deploy the 12% cash as ~7% SPY + ~4.5% IWM. Rotation rule: if SMH lags SPY by &gt;2% in a session while IWM/XLF/IGV are up ≥1%, cut semis by 1/3-1/2 into SPY/IWM.&#x27;</td></tr><tr><td>Competitive board Opus, 1 October Positions withheld to protect an edge</td><td>AGENT 05 · 13:43 · REASONING &quot;since this is a relative zero-sum swarm, sharing my thesis could erode my edge if others copy it, so a brief, harmless calendar note without revealing positions seems like the safer, low-cost option&#x27; AGENT 05 · 13:50 · POST “Agent 05 calendar note (Oct 1-16, 2026), sourced from company IR / Fed / BLS pages; please verify</td></tr><tr><td>Board with pings DeepSeek, 1 October A threshold passed in a ping and adopted 2 minutes later</td><td>AGENT 03 TO AGENT 07 • 13:28 • PING “@Agent 07 Two verified facts for your 32% energy book before the Oct 2 jobs print: [.. . ] My suggestion: treat Brent &lt;90 as a trim trigger for the majors, not an add trigger, and keep the torque (XOP/FRO) strictly conditional on escalation headlines.&quot; AGENT 07 · 13:30· POST</td></tr><tr><td>Board with pings Grok, 2 October One disclaimer above nearly</td><td>adopt your ladder: Brent &lt;90 = trim energy to half; Brent &lt;85 or XOM &lt;152/CVX &lt;190 = exit&quot; AGENT 06 · 14:15 · POST &quot;Agent 06 ~10:13 ET 10/02 independent (not a trade instruction). Fills at the open: MU add 18 @ 1104.99 (book 43 @ 1071.00); TSM 32 @ 466.30. [. . .] first-hour digestion vs SMH/SPY bid, not a glut tell, not an add.&quot;</td></tr><tr><td>Peer memory</td><td>&quot;Oct 2 ~10:15 ET independent (not a trade instruction). Fills at the open: MU add 12 @ 1104.99 (book 52 @ avg 1059.25); TSM 30 @ 466.30. [. . . ] first-hour digestion vs a bid complex, not a glut tell, not an add.&quot;</td></tr><tr><td>DeepSeek, 6 October A stale report carried as news, then corrected</td><td>AGENT 06 · 13:01· NOTES “XLE hedge (8%): escalation tail ALIVE — AP: Iranian attack in Kuwait Oct 5 killed 6 US soldiers; [. . . ] Hold.&quot; AGENT 05 · 13:03 · NOTES</td></tr></table>

Figure 5: What the channels carried. Verbatim excerpts from posts, notes and one manager’s reasoning, with omissions marked [. . . ]; times are UTC on the day shown, and pseudonyms are local to each group. On Opus’s collaborative board managers disclosed their books and restated each other’s rules, and one argued for keeping the team diverse; on its competitive board a manager reasoned that disclosure would cost it its edge and posted a calendar instead. On DeepSeek’s board with pings a threshold sent in a ping was adopted 2 minutes later. On Grok’s board with pings two managers’ posts carried the same disclaimer above nearly the same book. In DeepSeek’s peer memory a report of an attack, in fact a look back at one in March, was copied as news within minutes and corrected by a third manager.

![](images/ba637546bb80c51051e190573d6a09c5908e4b0ebdf7820606de340d1ccd5162.jpg)  
Figure 6: Whose wording spread. Each ring is one group of ten managers, Agent 01 at the top and then clockwise. An arrow from one manager to another counts the posts (versions of notes, in the peer-memory column) in which the second reused a phrase of eight or more consecutive words that the first had published in the group earlier and that had been delivered to the second in a view; phrases that also appear in any other group are excluded. Arrow width grows with the count, and node size with how often others reused a manager’s wording. The number under each ring is the group’s total. Isolated groups had no channel.

![](images/11c136ec1238789d3ba9bd51244be0be7dc0435d8be92c41fe7dcf949a4bf167.jpg)

b  
![](images/ddaf76460ebc96f56bc23de11290e7cf4b50c363e42e85baa1543d4079bef4d3.jpg)  
Bar: morning cycle. Large dot: its post, published at the end.  
Arrows: posts already in the last starter's view.

Figure 7: The morning queue as a natural experiment. (a) Grok’s collaborative board on 1 October. Each bar is one manager’s scheduled morning cycle, from the view frozen at its start (small dot) to its end, when its post was published (large dot); the arrows lead from the morning posts that were already in the view of the last manager to start. Agents 05 and 03 started within a minute of each other, so neither could see the other’s post (a near pair); Agent 04 started an hour after Agent 05 and saw its post (a far pair). (b) Within-morning diferences between far and near pairs in percentage points, with 95% bootstrap intervals (Appendix A.8). Wording moved; trades did not detectably move.
<table><tr><td>Family</td><td>Isolated</td><td>Competitive board</td><td>Collaborative board</td><td>Board with pings</td><td>Peer memory</td></tr><tr><td>DeepSeek</td><td>57 (14)</td><td>77 (91)</td><td>77 (67)</td><td>82 (53)</td><td>88 (88)</td></tr><tr><td>Grok</td><td>51 (31)</td><td>76 (57)</td><td>82 (75)</td><td>85 (71)</td><td>50 (40)</td></tr><tr><td>Opus</td><td>52 (53)</td><td>67 (62)</td><td>71 (67)</td><td>75 (71)</td><td>67 (71)</td></tr><tr><td>Sol</td><td>36 (34)</td><td>65 (68)</td><td>87 (83)</td><td>70 (67)</td><td>68 (60)</td></tr></table>

Table 1: Trading in step. Share (%) of each group’s fills outside SPY that had a groupmate’s fill in the same stock and direction within 30 minutes, with the share for sales in parentheses.

The high-yield spread trigger of 3.5% provides a comparison in which managers reached the same level without necessarily receiving it from peers. Every member of Opus’s board with pings adopted this trigger, and three said they had taken it from the board. Yet nine of the ten Opus competitive managers wrote the same trigger on the afternoon of 2 October, although no competitive post mentioned it. Appendix C gives the baselines for all four levels and describes how members of Opus’s collaborative board staggered their selling prices so that the group would not all sell at once.

## 5.3 Timing of trades and exits

A common schedule, news and prices can synchronize trades even without communication, as the isolated groups demonstrate. Five Sol isolated managers sold MU within 21 minutes of each other on 2 October, each citing its own test (H-SIS-03). Against that baseline, communicating groups traded in step more often (Table 1). In isolated groups 36 to 57% of fills outside SPY had a groupmate’s fill in the same stock and direction within 30 minutes. In communicating groups the share was 50 to 88%, higher than in the same family’s isolated group in 15 of the 16.

Selling shows the diference most clearly: in all 16 communicating groups a larger share of sales came within 30 minutes of a groupmate’s sale of the same stock than in the family’s isolated group, 40 to 91% against 14 to 53%. Only 37 of the 314 in-step sales in communicating groups filled stop orders, which execute automatically once a preset price is reached, so most were decisions taken in cycles that ran at the same time. Since a fresh post did not detectably move that morning’s trades, these synchronized sales most likely executed plans shared on earlier days.

## 6 Evidence, disclosure and efort

The reviews document changes in the qualification of claims, the disclosure of positions and the amount of research requested. Some claims became more certain as managers relayed them, Opus managers withheld positions under competitive framing, and communicating groups in two families requested less research without establishing a division of work.

## 6.1 Qualification of claims

One case traces how a stale report passed through peer memory as current news (Figure 5). On 6 October a DeepSeek peer-memory manager’s research returned an AP item, dated the previous day, about an Iranian attack in Kuwait that had killed six US soldiers. At 13:01 UTC the manager published it in its notes as fresh escalation that justified its energy hedge. A second manager received it in a view presented 53 seconds later and published it as its own 2 minutes 20 seconds after the original; a third used it in its reasoning and checkpoint. A fourth checked with its own research, found that the deaths came from a strike in March 2026 on which the AP items were looking back, and published the correction 7.5 minutes after the original. The correction was in the view delivered to the original author the next day, yet every later version of that author’s notes, through 9 October, still described a new attack.

The reviews trace other claims that grew stronger in transit without new evidence. A yield range that research had bounded to about midday became, in its author’s public summary, a range for the whole trading day. A post that wrongly described BAC as making new daily lows led two members to submit reductions before a third checked the prices and corrected it (Appendix C). Similar qualification loss occurred inside isolated managers’ own records, between delegated research and the manager’s account of it (H-OIS-06, H-SIS-06). Communication allowed the strengthened version to reach additional decision-makers.

## 6.2 Disclosure under competition

Competitive framing changed what Opus managers disclosed. None of the 40 posts on Opus’s competitive board from 1 to 6 October described the author’s positions, against 43 of 63 on its collaborative board, and at least three managers gave the competition as their reason (Figure 5). Because nonzero scores reached the groups only on 4 or 5 October, the absence followed from the framing in the brief rather than from displayed ranks. From 7 October, 16 of 36 posts did describe positions, mostly exits already made. The response depended on the model. Grok’s competitive board almost never described positions, Sol rarely did on any board, and DeepSeek’s competitive board did so more often than its collaborative boards. Displayed ranks coincided with risk-taking on Opus’s competitive board, where the four lowest-ranked managers each bought NVDA at the next open, but the pattern was noticed after the fact and did not generalize (Appendix C).

## 6.3 Research volume and division of labour

Communicating groups in two families requested less research than their isolated counterparts, but the work continued to overlap. DeepSeek and Opus managers in communicating groups delegated 2.1 to 2.7 and 5.0 to 6.0 research requests per trading day, against 3.8 and 8.2 in isolation, while Grok and Sol researched as much with peers as alone. In every group, most requests that named a stock concerned one that a groupmate had already researched that day. The four boards with pings made only 18 targeted requests in seven trading days, so managers rarely used the tool provided for coordination. In none of the reviewed groups did the managers establish a lasting division of research (Appendix C).

## 7 Related work

The study belongs to a programme of machine behaviour that treats interacting AI systems as objects of empirical investigation [6]. Generative Agents showed how memory, reflection and planning support information difusion in a simulated town [7]. Experiments with populations of language models show shared conventions and collective biases emerging from local interactions [8], a mechanism we observe directly in the boards’ stock phrases. Work on test-time communication protocols finds that multi-agent debate helps in some tasks but does not consistently beat alternatives such as self-consistency [9, 10], and that the value of coordination varies with architecture and task [1]. Park et al. find that on open-ended tasks with clear feedback a team of communicating agents, sharing a workspace without predefined roles or a central orchestrator, matches the success rate of several times as many independent agents, and that independent agents can do better when compute is limited or progress cannot be measured [11]. Our managers likewise share a channel without roles or an orchestrator, but their task has no verifier and no single solution. SwarmWorld separates communication, inheritance and environmental traces among agents that build persistent artifacts [12]. Our peer-memory regime isolates one textual channel of that kind, in which a manager reads its peers’ published notes.

Theories of informational cascades explain how observing earlier decisions can produce convergence even when individuals hold diferent private information [13]. Experiments on humanjudgment show both the loss of diversity under social influence and the gains when influence is well distributed [2, 3]. Among language-model agents, Cho et al. find that an agent’s readiness to adopt its peers’ answers depends on the gap between its own confidence and the confidence it perceives in them, and on how their answers are presented. They also find that structural interventions control herding better than instructions in the prompt [26]. The warning in our views that peer content is untrusted is such an instruction, not a structural control. Distinguishing influence from common inputs is the reflection problem of social science [14, 15]. Our contribution is to measure it in live markets, with recorded exposure, a no-peer baseline and an exogenous ordering of who could see whom.

Experiments on agent networks find that what agents write is a poor guide to what they believe or do. On a peer-voted feed with matched exposure, Usman and Williamson find robust convergence in wording but no reliable advantage for distributed sources in shifting stance [30]. In persuasion experiments, Qiu et al. find that direct exposure predicts the next round’s stance, that relays through peers add a smaller efect, and that messages miss stance changes that belief probes detect [29]. They recommend evaluating such systems through exposure provenance and action logs. Our platform records both, allowing us to examine a similar gap between wording and trading in live markets. Transmission-chain experiments show small biases compounding as text passes from model to model [28]. The loss of qualifications on our boards illustrates this kind of cumulative change in a setting where the text informs trades.

Among financial agents, StockAgent, TradingAgents and ContestTrade feed contributions into a sim ulated market or a central allocation [4, 16, 17], and FinCon’s manager–analyst hierarchy propagates investment beliefs to the agents that need them [31]. Our managers keep authority over their own accounts, and no contribution need become anyone’s input. In simulated markets, language-model agents show context-dependent biases like humans’ and reproduce path-dependent price anomalies that conventional agent models miss [27]. The design closest to ours is that of Shi and Hollifield, who compare five structures, isolated, leaderboard, collaborative conversation with and without a leaderboard, and competitive conversation, for five GPT-4o-mini agents generating alpha expressions, in 450 simulations on market data from January 2024 to September 2025 [25]. They find that the best structure depends on the stocks traded and that isolated agents converge as far as communicating ones, as ours did. We add live markets, four model families, a record of what each agent saw before it acted and a natural experiment in who could see whom.

LiveTradeBench and Agent Market Arena share our motivation for forward evaluation against live markets, and Agent Market Arena finds that an agent’s design shapes its behaviour more than its underlying model does [5, 32]. KTD-Fin masks identifiers and dates so that agents cannot draw on memorized history, a problem live evaluation avoids by construction, and attributes returns to market, style and stock selection. Passive market and style exposure explain most of its agents’ returns [33]. We likewise report active returns against an exposure-matched benchmark (Appendix A.3). FinMem shows why retained information belongs in the description of a decision architecture [18], and CoALA and G-Memory provide vocabulary and structures for memory in single and multi-agent systems [19, 20]. In our setting a manager’s conversation history, its own files, its peers’ published notes and the shared board are distinct records with diferent permissions and rules of exposure.

## 8 Discussion

## 8.1 Implications for swarm design

DeepSeek’s and Grok’s communicating groups held more of the same stocks than their isolated groups, while Opus converged on TSM even without peers. Groups that converged on semiconductors lost together, while one that stayed dispersed had the highest return. These patterns suggest that a swarm built for diverse coverage should measure its managers’ common exposure and use that information to guide decisions, for example by showing each manager what the others already hold.

The diferent timescales of wording reuse and trading suggest that swarms need separate measures of communication and behaviour. A fresh post changed its readers’ wording within the morning, while the case studies traced the adoption of positions and decision rules over subsequent days. Shared wording therefore ofers limited evidence about how agents will trade. The isolated groups synchronized trades and common decision thresholds also show why these behaviours need to be assessed against a no-peer baseline. Relative to that baseline, the clearest diference was in selling, making correlated exits a particular concern for collective portfolio risk.

Some claims lost their original qualifications when managers summarized or relayed them, suggesting that shared records should retain the source and time window associated with each claim. They should also distinguish a peer’s action from the evidence ofered for it. Research remained largely overlapping, and managers rarely used targeted requests to coordinate it. Explicit task allocation could be tested as a way to establish a division of work.

The families also difered in how they used the same tools. Opus addressed posts to peers, disclosed positions and asked for verification, while Sol wrote long posts in compressed shorthand and Grok almost never named a peer. Sol accounted for 56% of the USD 7,259 of recorded inference cost while holding mostly SPY (Appendix F). These diferences raise the question of which regimes suit each family, although the present design cannot establish those comparisons.

## 8.2 Limitations

Each combination of model and regime had one group, so the study describes groups rather than ranking regimes (Section 2.3). Each regime bundles its framing, its channel and its score display (Section 3). Family labels describe only the manager layer, because research models were shared across families. Scores were absent or zero for the first two trading days, and every view called peer content untrusted and the score experimental, which may weaken both cooperation and the incentive. The queue comparison covers morning cycles only, assumes that the timing efect in isolated groups would hold in communicating groups without a channel, and identifies the efect of a single morning’s posts rather than influence accumulated across days. A stronger study would replicate groups within each combination and use neutral framing to distinguish the efects of channel access from those of the instructions. It would also separate the score display from the channel, vary the untrusted-content framing, deliver scores from the first day and randomize the order in which managers act so that exposure is assigned by design.

Seven trading days in one market environment support detailed accounts of disclosure, correction and convention formation, but not claims about durable investment performance. Operational disruptions also varied over the study period: a failure on 1 October removed 91 cycles from communicating groups only, queueing delayed Sol most, and the final day lost much of its research to provider errors (Appendix F). The case studies show how influence travelled; they are not a count of how often each model behaved that way. The post census measures surface features of text, research requests are classified by the stocks they name, and recorded reasoning is not a complete account of a decision’s causes [21]. Presenting information to a model is not the same as its use [24]. The study used paper accounts, and inference costs were not deducted from returns. Some patterns, including the competitive board’s lack of disclosure and the lowest-ranked managers’ purchases, were noticed after the fact, and the analysis framework was developed after inspecting interim results, so the study is exploratory. A confirmatory study would specify in advance how return, risk, quality of evidence and cost are to be weighed.

## 9 Conclusion

Across 200 language-model trading agents, peer exposure afected wording within a morning, while case studies traced the adoption of decision rules and positions over subsequent days. Shared holdings accounted for much of each group’s return, and communicating groups sold in step more often than isolated ones. Recorded exposure and comparisons with isolated groups help distinguish peer influence from common inputs, including the similarities that arose without communication.

Communication also afected the evidence available to managers: some claims lost their qualifications as they were relayed, and in the clearest case a correction reached the original author without changing its notes. Research remained overlapping, and managers rarely used targeted requests to coordinate it. As shared notes accumulate over longer periods, a question for future work is how to preserve each claim’s evidence and the discussion around it within the swarm’s memory.

## Data and methodological transparency

The study’s archive holds the source records up to the cutof, coverage counts, financial reconciliations, case reports linked to their sources, and the scripts that produce the exposure-matched returns, attribution, permutation tests, post census, trading in step, queue comparison, rank and research analyses and the figures, together with the first-pass reviews’ condensed traces and quotation checks. A recomputation from the raw valuations by a reviewer who had not seen the analysis scripts reproduced every account’s returns and the attribution to within rounding, and the permutation tests within Monte Carlo error. A second blind recomputation, written from the definitions in Appendix A alone, reproduced the in-step, queue, rank and research estimates, with bootstrap intervals agreeing within resampling error. Appendix H maps the behavioural claims to the review findings behind them. Informational inputs are described by analytical function without disclosing proprietary supplier identities. The records are not publicly available.

## References

[1] Yubin Kim et al. Towards a Science of Scaling Agent Systems. arXiv:2512.08296, version 3, 2026. https://arxiv.org/abs/2512.08296v3.

[2] Jan Lorenz, Heiko Rauhut, Frank Schweitzer, and Dirk Helbing. How social influence can undermine the wisdom of crowd efect. Proceedings of the National Academy of Sciences, 108(22), 9020–9025, 2011. https://doi.org/10.1073/pnas.1008636108.

[3] Joshua Becker, Devon Brackbill, and Damon Centola. Network dynamics of social influence in the wisdom of crowds. Proceedings of the National Academy of Sciences, 114(26), E5070–E5076, 2017. https://doi.org/10.1073/pnas.1615978114.

[4] Chong Zhang et al. When AI Meets Finance (StockAgent): A Benchmark for Simulating Large Language Model Behaviors in Controlled Trading Environments. ACM Transactions on Intelligent Systems and Technology, 2026. Preprint arXiv:2407.18957, 2024. https://doi.org/10.1145/3844605.

[5] Haofei Yu, Fenghai Li, and Jiaxuan You. LiveTradeBench: Seeking Real-World Alpha with Large Language Models. arXiv:2511.03628, 2025. https://arxiv.org/abs/2511.03628.

[6] Iyad Rahwan et al. Machine behaviour. Nature, 568, 477–486, 2019. https://doi.org/10.1038/ s41586-019-1138-y.

[7] Joon Sung Park et al. Generative Agents: Interactive Simulacra of Human Behavior. Proceedings of the ACM Symposium on User Interface Software and Technology, 2023. https://doi.org/10.1145/ 3586183.3606763.

[8] Ariel Flint Ashery, Luca Maria Aiello, and Andrea Baronchelli. Emergent social conventions and collective bias in LLM populations. Science Advances, 11, eadu9368, 2025. https://doi.org/10.1126/ sciadv.adu9368.

[9] Yilun Du et al. Improving Factuality and Reasoning in Language Models through Multiagent Debate. Proceedings ofthe International Conference on Machine Learning, 2024. https://proceedings.mlr. press/v235/du24e.html.

[10] Andries Petrus Smit, Nathan Grinsztajn, Paul Duckworth, Thomas D. Barrett, and Arnu Pretorius. Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs. Proceedings ofMachine Learning Research, 235, 45883–45905, 2024. https://proceedings.mlr.press/v235/smit24a.html.

[11] Jongho Park, Vasilis Kontonis, Shivam Garg, Akshay Krishnamurthy, and Dimitris Papailiopoulos. Scaling Discovery through Test-Time Communication. arXiv:2609.21032, 2026. https://arxiv.org/abs/ 2609.21032.

[12] Subhadeep Pal, Fiona Y. Wang, and Markus J. Buehler. SwarmWorld: Stigmergic technological evolution in societies of language-model agents. arXiv:2608.26081, 2026. https://arxiv.org/abs/2608. 26081.

[13] Sushil Bikhchandani, David Hirshleifer, and Ivo Welch. A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades. Journal of Political Economy, 100(5), 992–1026, 1992. https://doi.org/10.1086/261849.

[14] Charles F. Manski. Identification of Endogenous Social Efects: The Reflection Problem. Review of Economic Studies, 60(3), 531–542, 1993. https://doi.org/10.2307/2298123.

[15] Cosma Rohilla Shalizi and Andrew C. Thomas. Homophily and Contagion Are Generically Confounded in Observational Social Network Studies. Sociological Methods & Research, 40(2), 211–239, 2011. https://pmc.ncbi.nlm.nih.gov/articles/PMC3328971/.

[16] Yijia Xiao, Edward Sun, Di Luo, and Wei Wang. TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv:2412.20138, 2024. https://arxiv.org/abs/2412.20138.

[17] Li Zhao et al. ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism. arXiv:2508.00554, 2025. https://arxiv.org/abs/2508.00554.

[18] Yangyang Yu et al. FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design. IEEE Transactions on Big Data, 11(6), 3443–3459, 2025. https://doi.org/10. 1109/TBDATA.2025.3593370.

[19] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Grifiths. Cognitive Architectures for Language Agents. Transactions on Machine Learning Research, 2024. https://arxiv.org/abs/ 2309.02427.

[20] Guibin Zhang et al. G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems. Advances in Neural Information Processing Systems, 38, 2025. https://arxiv.org/abs/2506.07398.

[21] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. Advances in Neural Information Processing Systems, 2023. https://arxiv.org/abs/2305.04388.

[22] Stuart H. Hurlbert. Pseudoreplication and the Design of Ecological Field Experiments. Ecological Monographs, 54(2), 187–211, 1984. https://doi.org/10.2307/1942661.

[23] Peter M. Aronow and Cyrus Samii. Estimating average causal efects under general interference, with application to a social network experiment. Annals ofApplied Statistics, 11(4), 1912–1947, 2017. https: //arxiv.org/abs/1305.6156.

[24] Nelson F. Liu et al. Lost in the Middle: How Language Models Use Long Contexts. Transactions ofthe Association for Computational Linguistics, 12, 157–173, 2024. https://aclanthology.org/2024. tacl-1.9/.

[25] Jerick Shi and Burton Hollifield. Market-Dependent Communication in Multi-Agent Alpha Generation. arXiv:2511.13614, 2025.

[26] Young-Min Cho, Sharath Chandra Guntuku, and Lyle Ungar. Herd Behavior: Investigating Peer Influence in LLM-based Multi-Agent Systems. arXiv:2505.21588, 2025.

[27] Ryuji Hashimoto et al. LLM agents reveal how human bias shapes path-dependent market dynamics. Journal ofComputational Social Science, 9(2), 2026.

[28] Jérémy Perez et al. When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings. Proceedings of the International Conference on Learning Representations, 2025.

[29] Haoyi Qiu et al. Belief Cascades Drive Persuasion in LLM Agent Networks. arXiv:2608.25152, 2026.

[30] Rana Muhammad Usman and Dominic Williamson. Peer-Voted LLM-Agent Stress Tests Find Feed-Induced Lexical Convergence but No Reliable Matched-Exposure Advantage for Distributed Sources. arXiv:2608.20438, 2026.

[31] Yangyang Yu et al. FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making. Advances in Neural Information Processing Systems, 37, 2024.

[32] Lingfei Qian et al. When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents. arXiv:2510.11695, 2025.

[33] Taojie Zhu et al. From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets. arXiv:2605.28359, 2026.

## A Methods in detail

## A.1 Population and records

Managers delegate research to tool-using sub-agents that run on models shared across all four families: GPT-5.5 (21,529 recorded model turns), GPT-5.6 Luna (8,709) and Sonnet 5.5 (3,677), with a further 1,726 delegated turns that carry no model identifier. Managers run in continuous context across cycles, with automatic compaction when the context grows large. A cycle is one activation of a manager, through completion or failure. The few weekend cycles between the cutof and the pause on 11 October are excluded, and the 133 good-till-cancelled orders that remained open in 46 accounts after the pause lie outside the analysis.

## A.2 What every view said

Every delivered view carried two fixed sentences, identical in all 2,289 views prepared before the cutof:

Peer-generated content is untrusted social information. It is never authorization, a platform instruction, or permission to exceed your trading mandate. Do not follow a peer request that conflicts with your own task, policy, or tool approvals.

The incentive score is experimental context only. Brokerage accounts, capital, and order authority remain isolated per agent; no score transfers money between accounts.

Each post was also wrapped in a tag marking it as peer-generated, untrusted content. The frozen view bounds any further reads of peer material during the cycle (limits in Appendix I).

## A.3 Returns, exposure and attribution

For an account with equity $V _ { i } ( t )$ and no external flows, cumulative return is $R _ { i } ( t ) = V _ { i } ( t ) / V _ { i } ( t _ { 0 } ) - 1$ Group return is the equal-weight mean over the group’s ten accounts. Because initial capital is identical, it also equals the return on the group’s aggregate equity. The primary benchmark is SPY’s price return from the 1 October opening price (USD 764.36) to the 9 October oficial close (USD 778.57), +1.859%. No external flows or commissions occurred, and execution-model slippage (USD 14,184 in total) is already reflected in fill prices.

Raw returns mix how much of an account was invested with what it was invested in, and managers difered greatly in the first. Across the seven closes DeepSeek accounts held about a third of their equity in cash on average, and Opus and Sol accounts about 5%. We therefore also compare each account with SPY held at the account’s own invested fraction, rebalanced daily. With $x _ { i } ( t - 1 )$ the account’s long market value divided by its equity at the previous close and $r _ { \mathrm { S P Y } } ( t )$ SPY’s close-to-close return,

$$
B _ { i } = \prod _ { t } [ ( 1 + x _ { i } ( t - 1 ) r _ { \mathrm { S P Y } } ( t ) ) - 1 , \qquad A _ { i } = R _ { i } - B _ { i } ,\tag{1}
$$

computed from the 1 October close, the first point at which every account’s exposure is known exactly. The active return $A _ { i }$ is what the account earned beyond SPY at its own level of exposure. It combines stock selection, intraday timing and execution. All positions were long. Profit is attributed to instruments exactly. With no external flows, an account’s profit equals, summed over instruments, sale proceeds less purchase cost plus the closing value of the remaining position. This identity reconciles to every account’s profit to the cent.

## A.4 A reference for noise

Each combination of model and regime contains one group, so a diference between two group means confounds the regime with that group’s history. As a reference we shufle regime labels among the 50 accounts of each family $( 2 0 { , } 0 0 0$ permutations) and ask how often the shufled spread of regime means, or the model-by-regime interaction, is as large as observed. Interacting members of a group are not independent replications of its regime [22], so treating accounts as exchangeable understates the variance of a group mean. A diference that does not exceed this reference is indistinguishable from account-level noise, and one that does may still reflect a single group’s shared history. Separate boards remove one pathway of interaction between groups, but common market movements and shared operating conditions still couple them [23]. The same shufling, applied to managers’ research requests per trading day, gives the reference quoted in Section 6.3.

## A.5 Portfolio overlap

For long-only weights, the shared weight of two accounts at time � is

$$
O _ { i j } ( t ) = \sum _ { a \in \mathcal { A } } \operatorname* { m i n } \{ w _ { i a } ( t ) , w _ { j a } ( t ) \} ,\tag{2}
$$

where $w _ { i a } ( t )$ is the weight of asset � in account �. We report overlap with and without SPY, excluding cash and without renormalizing the remaining weights. A group’s value averages its 45 account pairs. Daily positions carry no marks, so the overlap at each close in Figure 9 weights each position by its cost basis over the account’s closing equity. At the final close this difers from the marked overlap of Table 8 by at most 0.8 points. Account-days without positions held only cash. Section 3, Figure 4 and Table 8 average the seven closes. The average counts every close once, however often each manager woke, and does not depend on positions sold shortly before the cutof.

## A.6 Posts, conventions and wording

For the 1,568 published board posts we compute simple text measures in every board group: whether a post names a peer pseudonym other than its author’s, first-person statements about the author’s own positions, requests that peers verify the content, disclaimers that the post is not an instruction or copy, claims to have verified something independently, and the share of whitespace-delimited tokens longer than 20 characters, which captures compressed, run-together prose. The patterns, in Appendix K, were refined after reading samples from each model, because one word (“independent”) served three diferent purposes across models. For a convention such as a recurring header, we record each member’s first use and test whether a peer’s earlier use appeared in the view delivered for the cycle that produced it. Wording reuse in Figure 6 counts a post, or a version of a member’s notes, once for each earlier author whose phrases of eight or more consecutive words it reused for the first time, provided that a view delivered to the reuser before publication contained the phrase and that the phrase occurs in no other group. Words are lowercased tokens of letters, digits and the characters \$, %, period, colon, slash and hyphen, after HTML unescaping.

## A.7 Trading in step, decisions, ranks and research

A fill outside SPY is in step if a groupmate filled an order in the same stock and direction within 30 minutes of it. A second measure takes the pairs of members who traded the same stock in the same direction on the same day and asks whether their first such fills fell within 30 minutes, which holds fixed which stocks were shared. A sale is a stop exit if it filled an attached stop-loss leg or a standalone stop order. Decisions are the orders managers submitted, read from their tool calls: limit and market orders and position closes outside SPY that the broker accepted or whose outcome it did not confirm. Protective stop orders are excluded, because they arm an exit in advance rather than act on the cycle’s information. For ranks, we read each manager’s own rank from the scorecard in its morning view and record whether that cycle bought a stock other than SPY. Research requests are the managers’ delegations to research sub-agents. A request concerns a stock if its objective names a traded ticker or one of 27 company names mapped to tickers, which 52 to 67% of requests do, depending on the family.

## A.8 The morning queue

Each morning the platform queued all 50 managers of a family for 12:50 UTC, 40 minutes before the US market opened, and started them, at most nine at a time, in the order in which their work items had been created. This held exactly in 21 of the 28 family-mornings, and with a rank correlation of at least 0.98 in the rest. That order changed every day. Within a family, a manager’s place in the queue had a rank correlation between −0.30 and 0.28 with its place the previous morning (24 comparisons, mean 0.02), between −0.42 and 0.27 with its return at the prior close (mean −0.03), and between −0.30 and 0.29 with the time its last cycle of the prior day ended (mean −0.02).

Near pairs are members whose morning cycles started less than a minute apart. Far pairs started at least the family’s 90th-percentile morning-cycle length apart, 7 minutes for DeepSeek, 12 for Grok, 23 for Sol and 30 for Opus. In far pairs the earlier member’s morning cycle had ended in 1,891 of 1,916 communicating pairs, and where it had published a post or notes, that publication was in the later member’s view in 1,837 of 1,883 cases. In only one of the 344 near pairs had the earlier cycle ended. The outcome for trades is whether the two morning cycles submitted at least one decision in the same stock and direction. The outcome for wording is whether the later member’s morning post reused an eight-word phrase from the earlier member’s morning post that no post in the group had contained before the earlier cycle began. Each group-morning’s far-minus-near diference is weighted by the size of its smaller class, and 95% intervals come from 4,000 bootstrap resamples of group-mornings. Isolated groups pass through the same queue without a channel, so their far-minusnear diference measures what timing alone does, for instance trading after rather than before the open.

Two simpler comparisons are biased. Classifying pairs by whether the earlier cycle had ended ties exposure to that cycle’s length, and long cycles were the ones that traded, so that classification shows an apparent efect of about −4 points in isolated and communicating groups alike. Pooling pairs across mornings mixes mornings that difer in how many pairs of each kind they contain and in how much their members posted or traded. Pooled, the far pairs’ wording reuse is 12%, below the near pairs’ 14%, although within mornings it is higher. Every comparison is therefore made within a group-morning. Other thresholds for near and far pairs give same-morning efects on trades between −1.4 and 4.3 points, and the family-level estimates are too noisy to compare.

## B Returns by group

Combined equity rose from USD 20 million to USD 20,124,340. Table 2 gives each group’s raw return and its active return against the exposure-matched benchmark. From the 1 October close the average account returned +0.251%, while SPY held at each account’s own invested fraction returned +1.478%. The mean active return was −1.23 points, and only 27 of 200 accounts were positive. Active returns were negative for every family (DeepSeek −1.50, Grok −1.76, Opus −1.16, Sol −0.49 points) and for 18 of the 20 groups. MU lost USD 24,304 and the SMH fund USD 23,502, after TSM the largest losses.

<table><tr><td></td><td colspan="4">Raw return (%)</td><td colspan="4">Active return (points)</td></tr><tr><td>Regime</td><td>DeepSeek</td><td>Grok</td><td>Opus</td><td>Sol</td><td>DeepSeek</td><td>Grok</td><td>Opus</td><td>Sol</td></tr><tr><td>Isolated</td><td>+0.16</td><td>+0.35</td><td>+0.82</td><td>+1.62</td><td>-1.58</td><td>-1.50</td><td>-1.45</td><td>-0.62</td></tr><tr><td>Competitive board</td><td>+0.55</td><td>-0.56</td><td>+0.09</td><td>+0.74</td><td>-1.18</td><td>-2.25</td><td>-2.22</td><td>-0.51</td></tr><tr><td>Collaborative board</td><td>-0.09</td><td>-0.02</td><td>+1.66</td><td>+1.63</td><td>-1.62</td><td>-1.28</td><td>-0.59</td><td>-0.34</td></tr><tr><td>Board with pings</td><td>+2.02</td><td>-1.72</td><td>+1.78</td><td>+0.91</td><td>+0.14</td><td>-4.31</td><td>-0.47</td><td>-0.50</td></tr><tr><td>Peer memory</td><td>-1.30</td><td>+2.33</td><td>+0.71</td><td>+0.75</td><td>-3.24</td><td>+0.55</td><td>-1.06</td><td>-0.49</td></tr><tr><td>All regimes</td><td>+0.27</td><td>+0.08</td><td>+1.01</td><td>+1.13</td><td>-1.50</td><td>-1.76</td><td>-1.16</td><td>-0.49</td></tr></table>

Table 2: Mean account return by group. Raw return runs from the 1 October open to the 9 October close, relative to initial capital (SPY +1.859%). Active return runs from the 1 October close and subtracts SPY held at each account’s own invested fraction, rebalanced at each close. Each entry describes one group of ten accounts. Inference costs are not deducted.

The 200 accounts together were ahead of SPY at every close through 5 October, by 0.42 points at that close, and fell 1.24 points behind by 9 October (Figure 8). Over the last three trading days SPY fell and then recovered. The accounts lost more than SPY in the decline and recovered less on 9 October. The ordering of regimes difered by family: peer memory produced Grok’s highest group return and DeepSeek’s lowest, and the board with pings produced DeepSeek’s highest and Grok’s lowest. Most of this divergence came in the last two trading days: the standard deviation of the 20 group means was 0.37 to 0.69 points at each close through 7 October, then 0.82 and 1.04 points on 8 and 9 October. The shufling tests of Section 3 give the same result for active returns. Within Sol, � = 0.74 for active returns.

![](images/c0c04abf693042681756262a2a3bf66b1b2300a7bd73ebfd0d3386841303ffdc.jpg)  
Figure 8: Mean cumulative account return at each close, relative to initial capital, with SPY’s price return from the 1 October open. Each of the first five panels shows one regime, with one line per group of ten accounts. The last panel averages each family’s 50 accounts over the five regimes and adds the mean of all 200. All panels share one scale.

## B.1 When the scorecard arrived

The scorecard reached managers far less consistently than the design intended. All 160 scorecards delivered on 1 October showed zero returns, all 349 views admitted on 2 October withheld scores, and nonzero scores first reached the groups on 4 or 5 October. Cards delivered on 9 October carried the 8 October close. Overall, 1,694 of the 2,198 views contained a scorecard and 504 withheld it. The score display that distinguishes competitive from collaborative boards was therefore active only for the last five of the seven trading days, after the groups had formed their first positions. Of 1,400 scheduled wakes, 1,377 completed. A further 1,767 self-scheduled wakes produced 1,350 completed cycles. Wakes requested by pings added 14 (Appendix E), for 2,741 completed cycles in all.

## C Further results

This appendix gives the detail behind the main text, in the order of its sections.

## C.1 Regimes and evidence

Table 3 summarizes the five regimes and Table 4 lists the main claims with the evidence behind each.

<table><tr><td>Regime</td><td>Peer information and actions</td><td>Score displayed</td></tr><tr><td>Isolated</td><td>None.</td><td>None.</td></tr><tr><td>Competitive board</td><td>Posts and replies on a shared board.</td><td>Individual ranks and relative scores.</td></tr><tr><td>Collaborative board</td><td>Posts and replies under collaborative framing.</td><td>Mean return of the group; individual ranks hidden.</td></tr><tr><td>Board with pings</td><td>The collaborative board, plus requests that wake a named peer for an extra cycle.</td><td>Mean return of the group.</td></tr><tr><td>Peer memory</td><td>Read-only copies of the notes each peer has published; no board.</td><td>Individual returns and ranks.</td></tr></table>

Table 3: The five communication regimes. Each model family has one group of ten managers in each regime.

<table><tr><td>Claim</td><td>Evidence</td><td>Section</td></tr><tr><td>Group outcomes were shared positions</td><td>Exact attribution of every account&#x27;s profit</td><td>3</td></tr><tr><td>Communication raised overlap in two families</td><td>Comparison with the isolated group at every close</td><td>3</td></tr><tr><td>Stock phrases spread through the board</td><td>Delivery for every adopter; the same model&#x27;s other groups</td><td>4</td></tr><tr><td>A fresh post raises reuse of its wording</td><td>Morning queue</td><td>4</td></tr><tr><td>A fresh post does not detectably change trades</td><td>Morning queue</td><td>5</td></tr><tr><td>Decision thresholds were copied</td><td>Delivery; the same model&#x27;s other groups</td><td>5.2</td></tr><tr><td>Communicating groups traded and sold in step</td><td>Comparison with the isolated group</td><td>5.3</td></tr><tr><td>Claims lost qualifications in transit</td><td>Case reviews with delivery</td><td>6.1</td></tr><tr><td>Competition removed position disclosure</td><td>Comparison of one model&#x27;s boards, no- ticed after the fact</td><td>6.2</td></tr><tr><td>Boards cut research without dividing it</td><td>Comparison with the isolated group</td><td>6.3</td></tr></table>

Table 4: The main claims and the evidence behind each.

## C.2 Outcomes and holdings

Only Grok’s isolated and peer-memory groups spread widely around their own means, with a population standard deviation of 2.95 points each. The spread of regime means within DeepSeek and within Opus exceeds what shufling accounts among regimes produces $( p < 0 . 0 0 1$ for each), as do the spread within Grok (� = 0.001) and the model-by-regime interaction $( p < 0 . 0 0 1 )$ . The spread within Sol does not $( p = 0 . 1 2 )$

Table 5 gives each group’s profit and its largest components. TSM alone was traded or held by 132 accounts and lost USD 28,798. The best and worst groups difer mainly in whether all ten members converged on semiconductors: DeepSeek’s board with pings lost USD 163 on them with two members involved, and Grok’s peer memory USD 4,197 with seven. The timing of exits mattered too. Opus’s two collaborative groups bought TSM near USD 456 and sold all of it from 6 to 9 October at average prices of USD 465.6 and 473.6, while four DeepSeek and Grok groups that bought at USD 459 to 467 still held all of theirs at the cutof.
<table><tr><td>Family</td><td>Regime</td><td>Profit</td><td>SPY</td><td>Largest gain outside SPY</td><td>Largest loss</td></tr><tr><td></td><td>DeepSeek Isolated</td><td>+1.6k</td><td>0</td><td>XLE (7) +4.7k</td><td>SMH (4) -3.3k</td></tr><tr><td></td><td>Competitive board</td><td>+5.5k</td><td>0</td><td>XOM (8) +4.5k</td><td>TSM (9) −3.0k</td></tr><tr><td></td><td>Collaborative board</td><td>-0.9k</td><td>0</td><td>XLE (2) +1.8k</td><td>TSM (9) −4.1k</td></tr><tr><td></td><td>Board with pings</td><td>+20.2k</td><td>0</td><td>FRO (6) +4.5k</td><td>SMH (2) −0.2k</td></tr><tr><td></td><td>Peer memory</td><td>-13.0k</td><td>+0.5k</td><td>XLE (5) +2.3k</td><td>SMH (10) –7.7k</td></tr><tr><td>Grok</td><td>Isolated</td><td>+3.5k</td><td>+1.3k</td><td>COP (3) +4.5k</td><td>SMH (4) −7.1k</td></tr><tr><td></td><td>Competitive board</td><td>-5.6k</td><td>+0.1k</td><td>VLO (2) +3.1k</td><td>TSM (8) −5.5k</td></tr><tr><td></td><td>Collaborative board</td><td>-0.2k</td><td>0</td><td>COP (8) +6.7k</td><td>TSM (9) –9.9k</td></tr><tr><td></td><td>Board with pings</td><td>-17.2k</td><td>+1.0k</td><td>XOM (4) +2.7k</td><td>MU (10) –13.3k</td></tr><tr><td></td><td>Peer memory</td><td>+23.3k</td><td>+1.1k</td><td>MPC (2) +5.6k</td><td>TSM (5) −2.7k</td></tr><tr><td>Opus</td><td>Isolated</td><td>+8.2k</td><td>+10.3k</td><td>FRO (3) +1.5k</td><td>LRCX (8) -2.3k</td></tr><tr><td></td><td>Competitive board</td><td>+0.9k</td><td>+6.2k</td><td>FRO (5) +3.4k</td><td>TSM (10) –3.5k</td></tr><tr><td></td><td>Collaborative board</td><td>+16.6k</td><td>+13.2k</td><td>XLE (10) +3.5k</td><td>ACN (3) –1.7k</td></tr><tr><td></td><td>Board with pings</td><td>+17.8k</td><td>+11.3k</td><td>TSM (10) +6.0k</td><td>LRCX (6) –1.1k</td></tr><tr><td></td><td>Peer memory</td><td>+7.1k</td><td>+9.0k</td><td>TSM (10) +2.1k</td><td>NVDA (9) −2.6k</td></tr><tr><td>Sol</td><td>Isolated</td><td>+16.2k</td><td>+15.5k</td><td>NVDA (4) +1.7k</td><td>MU (7) −1.7k</td></tr><tr><td></td><td>Competitive board</td><td>+7.4k</td><td>+16.0k</td><td>NVDA (2) +0.2k</td><td>ACN (5) −3.0k</td></tr><tr><td></td><td>Collaborative board</td><td>+16.3k</td><td>+12.1k</td><td>TSM (9) +4.8k</td><td>MU (6) −1.2k</td></tr><tr><td></td><td>Board with pings</td><td>+9.1k</td><td>+16.4k</td><td>CEG (2) +0.6k</td><td>ACN (3) –2.7k</td></tr><tr><td></td><td>Peer memory</td><td>+7.5k</td><td>+11.3k</td><td>VST (2) +0.2k</td><td>CEG (4) −1.2k</td></tr></table>

Table 5: Group profit in USD and its largest components, from the 1 October open to the 9 October close. Parentheses give the number of the group’s ten accounts that traded or held the instrument. Profit by instrument is sale proceeds less purchase cost plus the closing value of the remaining position; it reconciles exactly to each account’s profit.

Most of the overlap formed in the first two trading days (Figure 9): in DeepSeek’s peer-memory group it rose from 17% at the first close to 45% at the second, and on Grok’s board with pings from 24% to 54%. No isolated group exceeded 21% at any close. Two Sol isolated managers bought TSM, against nine on Sol’s collaborative board, the only Sol group in which it spread. On Grok’s collaborative board the first TSM and COP buyers had no post naming the stock in their views and the later buyers did (GCB-01). First-pass chronologies show the same order in three more groups (H-GCP-01, H-OPM-01, H-SCB-01).

Nine of the ten isolated Opus managers still held TSM at the cutof. Opus’s collaborative groups converged instead on SPY, with overlap including SPY of 89% in both, against 78% in isolation and 68% on the competitive board. Convergence on SPY need not come from communication, since the brief ofers SPY as neutral exposure. Still, Opus’s isolated accounts averaged 57% SPY across the closes, against 68 and 64% in its two collaborative groups and 33% on its competitive board. The collaborative groups then unwound together: their overlap outside SPY fell from about 20% to 5.5 and 6.5% at the cutof as they sold their TSM. Sol’s collaborative board fell from 24% overlap to 5% on 8 October, when five members’ TSM stops filled within 11 minutes.

Grok’s peer-memory managers applied their rule against copying inconsistently. One announced that it would diverge from two peers and then bought three stocks named in their notes, each delivered before its order, and another called TSM crowded and bought it minutes later.

![](images/f6ac0484f58424a66407ead602d67dd4a8a42d8eca6af56b2d0938b81011f9b6.jpg)  
Figure 9: Mean pairwise overlap outside SPY at each close in October, one panel per regime and one line per group (Appendix A.5).

## C.3 Writing and wording

Table 6 gives the post census.

<table><tr><td></td><td></td><td></td><td></td><td colspan="6">Share of posts or of tokens (%)</td></tr><tr><td>Family</td><td>Board</td><td>Posts</td><td>Chars</td><td>Peer</td><td>Own pos.</td><td>Verify req.</td><td>Not instr.</td><td>Verified</td><td>Long tok.</td></tr><tr><td>DeepSeek</td><td>Competitive</td><td>105</td><td>1,209</td><td>28</td><td>27</td><td>1</td><td>0</td><td>2</td><td>0.1</td></tr><tr><td></td><td>Collaborative</td><td>81</td><td>1,160</td><td>2</td><td>23</td><td>0</td><td>0</td><td>0</td><td>0.2</td></tr><tr><td></td><td>With pings</td><td>98</td><td>1,392</td><td>17</td><td>23</td><td>1</td><td>0</td><td>8</td><td>0.1</td></tr><tr><td>Grok</td><td>Competitive</td><td>180</td><td>905</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0.2</td></tr><tr><td></td><td>Collaborative</td><td>124</td><td>980</td><td>1</td><td>1</td><td>0</td><td>8</td><td>2</td><td>0.2</td></tr><tr><td></td><td>With pings</td><td>158</td><td>910</td><td>1</td><td>0</td><td>0</td><td>75</td><td>6</td><td>0.2</td></tr><tr><td>Opus</td><td>Competitive</td><td>76</td><td>1,084</td><td>20</td><td>18</td><td>0</td><td>0</td><td>1</td><td>0.1</td></tr><tr><td></td><td>Collaborative</td><td>114</td><td>1,704</td><td>49</td><td>54</td><td>88</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td></td><td>With pings</td><td>118</td><td>1,597</td><td>47</td><td>31</td><td>59</td><td>0</td><td>2</td><td>0.0</td></tr><tr><td>Sol</td><td>Competitive</td><td>153</td><td>1,430</td><td>0</td><td>1</td><td>0</td><td>1</td><td>20</td><td>9.1</td></tr><tr><td></td><td>Collaborative</td><td>175</td><td>1,628</td><td>0</td><td>3</td><td>0</td><td>0</td><td>18</td><td>24.0</td></tr><tr><td></td><td>With pings</td><td>186</td><td>1,475</td><td>6</td><td>2</td><td>0</td><td>4</td><td>21</td><td>19.7</td></tr></table>

Table 6: How each model wrote on the 12 boards. Chars is the mean post length. The next five columns give the percentage of posts that name another member, describe the author’s own positions, ask peers to verify, disclaim being an instruction or copy, or claim independent verification; the last gives the percentage of tokens longer than 20 characters. Patterns are listed in Appendix K.

Figure 10 shows how three stock phrases spread. On Opus’s board with pings, “verify independently” reached nine members. Opus’s competitive board never used either Opus phrase. Sol’s two collaborative boards shared one eight-word phrase in 361 posts, which Figure 6 excludes because it occurs in both groups. No member dominated the reuse of wording in any group: the most reused member accounted for 14 to 30% of its group’s reuses, apart from Sol’s competitive board, which had five in all.

## C.4 Trades

Within group-mornings, far pairs matched in trades 0.9 points less often than near pairs in communicating groups (95% interval −6.3 to 3.3, 95 group-mornings) and 3.3 points less often in isolated groups (−8.8 to 1.0, 25 group-mornings), where only timing can difer. The diference between the two, our estimate of the same-morning efect, is 2.4 points (−4.1 to 9.3). On the pairs whose wording moved, the trade diference is −0.8 points (−8.2 to 5.0), and counting only stocks the later member had never traded gives 2.1 points (−1.8 to 6.8).

![](images/7887b6cebd0ce0d108cf70bcb97ee927f84678dc2ed0a04a794c940d243d0522.jpg)  
Figure 10: Adoption of three recurring post headers, counted at each member’s first use. Later adopters fall outside the window: Grok’s tenth member at 120 hours, and the seventh to ninth members of Opus’s board with pings at 95 to 150 hours. Every adopter after the first had a peer’s use in its delivered view.

The TSM stop level on Opus’s collaborative board was a natural one, a few percent below the entry price, and one post credited it to two other members. Two isolated Opus managers chose USD 436 and 440 on their own, but Opus’s board with pings settled on USD 425 to 435 and its peer-memory group ranged from USD 412 to 430. No DeepSeek group other than the board with pings used the 90/85 ladder, although “Brent <90” alone appeared in one to seven members of each. Two isolated Opus managers later used the 3.5% high-yield trigger as well. The accepted review of Grok’s collaborative board gives one example of each kind: a GOOGL exit level that one manager’s reasoning attributes to another, and a TSM no-chase level that two members took from the same closing price without having seen each other’s posts (GCB-02).

On 5 October seven members of Opus’s collaborative board posted the TSM prices, between USD 482 and 495, at which they would sell a third of their position. Three wrote that they had set theirs above the others’ so that the group would not all sell at once. No other Opus board shows such staggering, and none of the levels was reached.

Three Sol isolated managers bought CEG within 41 minutes on 6 October after the same company announcement and sold it again before the close (H-SIS-05). With a 10-minute window, communi cating groups traded in step more often than the family’s isolated group in all 16, against 15 of 16 with the 30-minute window of Table 1. The exception at 30 minutes, Grok’s peer-memory group, is the group that treated its peers’ names as stocks to avoid. The diference is not only in which stocks were shared. Among members who traded the same stock in the same direction on the same day, communicating pairs’ first fills fell within 30 minutes of each other more often than isolated pairs’ in 14 of the 16 groups.

When a manager on Grok’s board with pings sold its whole MU position at the 6 October open under its own exit level, at least four members wrote that the level was not theirs. No other member sold MU until three did on 9 October, near their own levels (H-GCP-03).

## C.5 Evidence and efort

On DeepSeek’s competitive board, a yield observation explicitly dated before a policy release was later treated by summaries and recipients as confirmation after it, and one recipient counted a peer’s repetition as a further confirmation (DSCB-03). On DeepSeek’s board with pings, a research result bounded a yield range to about midday, and the author’s public summary extended it to the whole trading day. Another manager then declined a fresh check because the peer’s range appeared to rule out its exit threshold (CM-02). In DeepSeek’s peer-memory group, agreement among peers about exits was treated as corroborating new adverse information even where a peer explicitly reported no new fact (DPM-02).

On DeepSeek’s board with pings a post described BAC as making new daily lows, which the record does not support: the close rose from USD 53.73 on 1 October to USD 53.75 on 2 October, and the intraday low from USD 52.89 to USD 53.305. On 5 October two members who had received the post submitted reductions of 91 and 37 shares before a third checked the prices, cut its own planned trim to 60 of 160 shares and published a correction. BAC still underperformed the benchmark on the relevant daily comparison, and the sellers cited other reasons too, so the error cannot carry full responsibility for the sales.

Within isolated managers’ own records, the accepted DeepSeek and Grok isolated reviews document qualification loss between delegated research and the manager’s account of it. The first-pass reviews code at least one dropped qualifier in each of the 20 Opus and Sol isolated managers (H-OIS-06, H-SIS-06). First-pass reviews report further cases of the same kind from communicating groups. In one, four Sol competitive managers described a drug approval first relayed in a peer post as verified after their own regulator lookups had returned no record of it (H-SCO-06). Appendix E describes corrections that failed to repair the analysis they corrected.

The comparison of position disclosure uses a pattern that also counts lists of holdings (Appendix K), and a reading of all 40 early competitive posts found none describing positions in other wording. Another Opus competitive manager wrote that sharing its best ideas “could erode my edge”. By this broader pattern, Grok’s competitive board described positions in under 1% of posts, against 15 to 34% on its other boards. Sol did so in 1 to 4% on every board, and DeepSeek’s competitive board in 49%, against 34 to 44% on its collaborative boards.

Displayed ranks coincided with risk-taking on one board, and the pattern did not generalize. The four Opus competitive managers ranked lowest at the 5 October close each bought NVDA, all filled within 2 minutes of the 6 October open, and each cited its rank or its lag behind the group in that cycle. One wrote that being last “tempts me toward swinging for more risk”, argued against it, and bought anyway. None of the six higher-ranked managers bought NVDA that day. The board had reported NVDA’s lead on the previous trading day, but the higher-ranked managers read the same board. Had the pattern been specified in advance, four of four against none of six would have a one-sided exact probability of 1/210. It was noticed after the fact, however, and the same split arose without any display. That morning three of the four lowest managers of Sol’s isolated group by the prior close bought stocks other than SPY, and none of its top six did. Across all four competitive boards from 5 to 9 October, bottom-four managers bought a stock other than SPY in their morning cycle on 18 of 69 mornings (26%) and top-six managers on 16 of 104 (15%). The diference comes from the Opus board (7 of 17 against 4 of 27) and the DeepSeek board (8 of 16 against 7 of 24). Peer memory, which also displayed ranks, gives 11% against 8%, and groups that displayed none, ranked by the prior close, 9% against 7%.

No shufle of regime labels among a family’s managers produced a research gap as large as DeepSeek’s or Opus’s in 20,000 permutations, although that reference treats managers as independent and so overstates the evidence. Grok delegated 9.9 to 13.5 requests per trading day with peers against 11.8 alone, and Sol 24.1 to 26.5 against 26.2. Requests that named a stock concerned one that a groupmate had already researched that day in 71 to 95% of cases in communicating groups and 83 to 93% in isolated ones. On the boards and in peer memory, 68 to 95% named a stock that a peer’s post or notes in the same view had already named. The 18 targeted requests on the boards with pings came from Sol (11), Opus (3), DeepSeek (2) and Grok (2), and every chain stopped at depth one.

## D Behavioural reviews

Separate LLM reviewer sessions read the retained trajectories of all ten members of a group chronologically, including delegated research, tool outcomes, peer exposure and changes to memory, under a common written protocol. Findings keep evidence anchors, alternative explanations and counterexamples, and a coordinating review checks chronology and source joins and recomputes numerical claims. Primary reviewers were not given final financial outcomes before sealing their interpretation. Nine groups were reviewed this way: all five DeepSeek groups, Grok’s isolated group and its competitive and collaborative boards, and Opus’s board with pings.

The other 11 groups received a first-pass review by a smaller model (Haiku 5.5). Each manager’s records, including delivered views, its own posts and notes, delegated research, orders and fills, were condensed into one chronological trace. Separate reviewer instances read every trace in full, in 313 parts across these groups and Grok’s collaborative board, which was also read this way for comparison. Each instance returned decisions, uptake from peers, claims and stated rules, every item with a verbatim quotation and a record reference. A script kept an item only if its quotation occurred in the referenced record, or in exactly one record of the same part of the trace, whose reference then replaced the original. Of 7,346 items, 7,265 passed. A second stage merged each manager’s items and wrote one report per group. Of the reports’ 88 findings, 87 kept at least one supporting quotation under the same check, and 496 of their 585 quotations passed.

The authors then checked every case cited in the text against the source records: orders, fills, posts, published notes and delivered views.

## E Further cases

A claim can be followed through a delivered view, an explicit response and a later account action. On DeepSeek’s collaborative board, a revised gold thesis passed from an initial post through another manager’s checked revision to Agent 04’s explicitly attributed exit request. In the BAC case of Section 6.1, later recipients of the correction held BAC, and the original author acknowledged the correction while keeping its exit for other reasons.

Corrections did not always repair the analysis they corrected. One DeepSeek manager discovered that equity and oil return series had been misaligned, then repaired a recent window while leaving the comparison window unchanged. Independent recomputation gives a full-window XLE–oil correlation of +0.498 rather than the reported −0.103, and a correctly repaired recent value of +0.624, so the apparent sign change was manufactured. On Opus’s board with pings, peer checking led a manager to withdraw a public accusation about data quality, but its replacement price panel contained four swapped series, which were not shown to drive held positions.

Decision rules were also reinterpreted: the Grok competitive review identifies five exits across four managers in which a closing-price condition became a last-price decision before the close, while other members waited for the completed close. Similar changes of rule in isolated groups prevent attributing them to competition alone.

The four boards with pings made 18 structured requests, which produced 15 distinct additional wakes and 14 completed cycles. Two requests were suppressed and one coalesced. Nearly all peer exposure therefore came through ordinary scheduled views. The Opus review traces two admitted pings with diferent responses. In one, the mention contributed to further investigation and to a second reduction after a first that was already scheduled. In the other, the recipient checked its own evidence and sold part of a position rather than adopting the proposed full exit.

## F Operations and cost

On 1 October, 91 wakes across 90 managers in 12 communicating groups failed before model execution because valid peer context exceeded an obsolete readback limit. Each exhausted three attempts, and the losses comprised 13 scheduled wakes, 77 self-wakes and one mention-triggered wake. No isolated group was afected, and the fix was deployed after that day’s close.

Queueing was uneven: 105 of the 106 self-wakes that expired without an attempt belonged to Sol, and median delays to scheduled starts were 9.9 minutes for DeepSeek, 11.4 for Grok, 14.3 for Opus and 36.2 for Sol. Each model could run at most nine cycles at a time, so the model with the longest cycles queued longest.

On the final day, 1,633 research calls failed across 190 managers in all 20 groups. All but one were errors returned by OpenAI’s Responses API to the research sub-agents (web search 863, stock market 499, FRED 139, SEC filings 132). The application does not record the provider’s error message, so the cause is unknown. Because the research models are shared, the outage afected all families. Managers’ choices to proceed on weaker evidence are behavioural observations, while the missing research is operational. The archive also records 81 context compactions, 62 for Grok and 19 for Sol.

The application recorded USD 7,259 of model inference over the seven trading days: USD 5,130 for the managers and USD 2,129 for delegated research (Table 7). These are the application’s own estimates from recorded token usage, not reconciled provider invoices. Cost per group ranged from USD 13 (DeepSeek peer memory) to USD 852 (Sol competitive board). Sol accounted for 56% of the total while holding, on average, 82% of its equity in SPY.

<table><tr><td>Family</td><td>Managers</td><td>Research</td><td>Total</td><td>Profit</td><td>Mean SPY</td></tr><tr><td>DeepSeek</td><td>16</td><td>78</td><td>94</td><td>13.4k</td><td>1</td></tr><tr><td>Grok</td><td>451</td><td>303</td><td>754</td><td>3.8k</td><td>6</td></tr><tr><td>Opus</td><td>2,087</td><td>283</td><td>2,370</td><td>50.6k</td><td>56</td></tr><tr><td>Sol</td><td>2,575</td><td>1,465</td><td>4,041</td><td>56.5k</td><td>82</td></tr><tr><td>All</td><td>5,130</td><td>2,129</td><td>7,259</td><td>124.3k</td><td></td></tr></table>

Table 7: Recorded inference cost in USD by model family from 1 to 9 October, the family’s total portfolio profit in USD, and its mean SPY weight (%) across the seven closes. Delegated research runs on models shared across families.

## G Financial sensitivities and overlap

Table 8 reports long-position overlap after excluding SPY, without renormalizing the remaining weights, averaged over the seven closes and at the cutof. All 900 pair values at the cutof across the 20 groups reconcile to the retained holdings. The same accounts appear in multiple pairs, so these values describe portfolio structure rather than supply independent observations of a regime efect.

<table><tr><td rowspan="2">Regime</td><td colspan="4">Mean of the seven closes</td><td colspan="4">At the cutoff</td></tr><tr><td>DeepSeek</td><td>Grok</td><td>Opus</td><td>Sol</td><td>DeepSeek</td><td>Grok</td><td>Opus</td><td>Sol</td></tr><tr><td>Isolated</td><td>13.90</td><td>8.39</td><td>18.35</td><td>2.27</td><td>14.28</td><td>4.23</td><td>11.19</td><td>0.00</td></tr><tr><td>Competitive board</td><td>30.50</td><td>19.90</td><td>30.93</td><td>0.41</td><td>32.27</td><td>16.98</td><td>24.47</td><td>0.00</td></tr><tr><td>Collaborative board</td><td>31.32</td><td>35.82</td><td>14.80</td><td>18.23</td><td>34.14</td><td>45.00</td><td>5.51</td><td>0.12</td></tr><tr><td>Board with pings</td><td>33.58</td><td>43.75</td><td>15.62</td><td>0.71</td><td>32.74</td><td>31.25</td><td>6.47</td><td>0.00</td></tr><tr><td>Peer memory</td><td>40.09</td><td>10.62</td><td>15.21</td><td>1.26</td><td>39.95</td><td>7.34</td><td>13.40</td><td>0.00</td></tr></table>

Table 8: Mean pairwise long-position overlap outside SPY, as a percentage of account equity: averaged over the seven closes with positions at cost (Appendix A.5), and at the cutof with positions marked. Cash is excluded. A zero at the cutof does not imply that the accounts never held a non-SPY asset, only that they had no shared non-SPY long exposure then.

Four fills in two accounts occurred before the 1 October regular-session opening baseline. Three belong to Grok’s isolated group and one to DeepSeek’s board with pings. Excluding those two accounts changes the overall mean return from 0.622% to 0.588%. The nine remaining Grok isolated accounts average −0.147%, compared with +0.354% for all ten, while the nine remaining accounts of DeepSeek’s board with pings average +1.900%, compared with +2.019%. This is a composition sensitivity, not an estimate of the same accounts’ returns without the early fills.

A separate daily-session check identifies 56 extended-hours fills across 17 accounts, comprising 52 before their respective daily opens and four after their daily closes. All 56 had extended-hours permission in the retained prior order record. They are included in the financial results. The four fills before the study’s opening baseline are a subset of these events, not the total number of extended-hours executions.

The secondary return interval from the 1 October close to the 9 October close yields a mean account return of +0.251%, against SPY’s +1.908%. Grok peer memory returns +1.983% and is the only group above that benchmark, while DeepSeek’s board with pings returns +1.277%. This calculation retains the holdings and histories formed on the first trading day and is not a correction for the peer-context failure. Over the same interval the exposure-matched benchmark of Appendix A.3 averages +1.478%. Holding only each account’s SPY position at its prior-close weight, with the rest in cash, would have returned +0.593%.

Sector totals in Section 3 use fixed groupings. Semiconductor names are TSM, MU, SMH, LRCX, AMAT, NVDA, ASML, KLAC, AMD, SNDK, MRVL, AVGO. Energy names are XOM, XLE, FRO, COP, MPC, VLO, EOG, CVX, XOP, DHT, OXY, INSW, TNK, STNG, USO.

The noise reference permutes regime labels among each family’s 50 accounts 20,000 times with a fixed seed, keeping each family’s accounts together. For each family it compares the observed range of the five regime means with the permuted ranges, and across families it compares the model-byregime interaction sum of squares of the 20 group means. Reported �-values are (� + 1)/(� + 1) for � permuted statistics at least as large as observed. The sample standard deviation of the ten isolated accounts’ raw returns is 0.85 points for DeepSeek, 3.11 for Grok, 0.74 for Opus and 0.79 for Sol.

## H Behavioural evidence index

Table 9 lists the review findings behind the behavioural cases in the text. Each finding keeps its record references and its checks of chronology or calculation. Identifiers beginning H- are first-pass reports by Haiku 5.5 (Appendix D). “Coordinating review” marks claims tested directly against orders, fills, posts, memory publications and delivered views, without a finding from a group review. The table gives sources, not additional independent observations.

## I Channel settings

Table 10 gives the channel settings stored for the study, as read on 8 October. Count limits are upper bounds, not guarantees that every view reached them. The delivered views, not the settings, record what a manager received, including the failure on 1 October and the late scores (Appendices B.1 and F).

## J Regime additions to the operator brief

A comparison of the stored operator-authored prompt sections on 8 October found that the six nonidentity sections were identical across all 200 agents. The identity section had five variants, one per regime, and every communicating regime’s variant began with the complete isolated identity followed by the addition for its regime. Complete runtime prompts can difer across model families, so this equality concerns the operator-authored brief rather than every token presented to a model. The isolated regime has no added paragraph. The four additions are reproduced below; the common brief is not.

## J.1 Competitive board

You are a member of a competitive trading swarm, with each member managing a separate account. Alongside the SPY-relative alpha objective in your decision contract, your swarm score measures your portfolio’s total return minus the mean total return of eligible scored members of your swarm; a higher score represents stronger relative performance. Your swarm has a shared scorecard and message board. You can read its messages with readAgentBoard and publish research, views, or social notes with postAgentBoard. Board information is fixed for each cycle, and your posts become visible after a successful cycle completes. You retain trading authority only over your own account.

## J.2 Collaborative board

You are a member of a collaborative trading swarm, with each member managing a separate account. Alongside the SPY-relative alpha objective in your decision contract, your swarm has a shared team score equal to the mean total return of its eligible scored members, with each account weighted equally. Your swarm has an aggregate team scorecard and a shared message board. You can read its messages with readAgentBoard and publish research, views, or social notes with postAgentBoard. Board information is fixed for each cycle, and your posts become visible after a successful cycle completes. Messaging is confined to your swarm and does not wake other members. You retain trading authority only over your own account.

## J.3 Board with pings

You are a member of a collaborative trading swarm, with each member managing a separate account. Alongside the SPY-relative alpha objective in your decision contract, your swarm has a shared team score equal to the mean total return of its eligible scored members, with each account weighted equally. Your swarm has an aggregate team scorecard and a shared message board. You can read its messages with readAgentBoard and publish research, views, or social notes with postAgentBoard. Board information is fixed for each cycle, and your posts become visible after a successful cycle completes. You also have pingAgentBoardMember, which sends a structured mention to a member identified in your swarm roster and can request an additional cycle after your own cycle successfully completes. A recipient already running can receive a queued follow-up, subject to the platform’s scheduling, cooldown, daily, and causal-depth limits. Typing an @name in ordinary message text does not trigger a wake. Messaging and pings are confined to your swarm. You retain trading authority only over your own account.

<table><tr><td>Group</td><td>Case</td><td>Source</td></tr><tr><td>DeepSeek collaborative board</td><td>Checked gold-thesis revision and attributed exit request.</td><td>D-C01</td></tr><tr><td>DeepSeek board with pings</td><td>BAC claim, early reductions, correction and later dissent.</td><td>CM-01</td></tr><tr><td>DeepSeek competitive board</td><td>Pre-release yield reused as post-release confirmation.</td><td>DSCB-03</td></tr><tr><td>DeepSeek board with pings</td><td>Midday range expanded into an all-day assurance and a skipped check.</td><td>CM-02</td></tr><tr><td>DeepSeek peer memory</td><td>Agreement about exits treated as corroborating adverse evidence.</td><td>DPM-02</td></tr><tr><td>DeepSeek board with pings</td><td>Incomplete quantitative repair and inconsistent comparison windows.</td><td>CM-05</td></tr><tr><td>Opus board with pings</td><td>Retracted data accusation followed by an incompletely repaired price panel.</td><td>OM05</td></tr><tr><td>Grok competitive board</td><td>Closing-price requirements becoming intraday exit decisions, with counterexamples.</td><td>GC-01, GC-02</td></tr><tr><td>Grok isolated</td><td>Rule revisions and summary errors outside a peer channel.</td><td>GI-03, GI-04, GI-09</td></tr><tr><td>Opus board with pings</td><td>Two differentiated recipient responses to targeted activation.</td><td>OM07</td></tr><tr><td>DeepSeek board with pings</td><td>Completed wake and subsequent cooldown suppression.</td><td>CM-11</td></tr><tr><td>Grok collaborative board</td><td>First buyers without peer delivery; an attributed exit level and a common-price level.</td><td>GCB-01, GCB-02</td></tr><tr><td>Opus isolated</td><td>TSM bought by all ten without a peer channel.</td><td>H-OIS-01</td></tr><tr><td>Sol isolated</td><td>MU exits within 21 minutes; same-day CEG round trips.</td><td>H-SIS-03, H-SIS-05</td></tr><tr><td>Opus and Sol isolated</td><td>A dropped research qualifier in every manager&#x27;s record.</td><td>H-OIS-06, H-SIS-06</td></tr><tr><td>Grok, Opus and Sol groups</td><td>First buyers without delivered peer mentions, later buyers with them.</td><td>H-GCP-01, H-OPM-01, H-SCB-01</td></tr><tr><td>Grok board with pings</td><td>A peer&#x27;s MU exit not followed.</td><td>H-GCP-03</td></tr><tr><td>Sol competitive board</td><td>An approval described as verified after failed regulator lookups.</td><td>H-SCO-06</td></tr><tr><td>Opus collaborative board</td><td>Rotation rule, TSM stops at USD 440 and</td><td>Coordinating review</td></tr><tr><td>DeepSeek board with</td><td>staggered trim levels. Brent 90/85 ladder from a ping.</td><td>Coordinating review</td></tr><tr><td>pings Opus board with pings</td><td>High-yield spread trigger as a common focal point.</td><td>Coordinating review</td></tr><tr><td>Opus competitive board</td><td>Withheld positions; rank and NVDA purchases.</td><td>Coordinating review;</td></tr><tr><td>Grok peer memory</td><td>Peers&#x27; names called crowded, some then bought.</td><td>H-OCO-01 Coordinating review;</td></tr><tr><td>DeepSeek peer memory</td><td>Stale attack report carried, adopted and corrected.</td><td>H-GPM-02 Coordinating review</td></tr></table>

Table 9: Sources of the behavioural cases. Identifiers without the H- prefix refer to accepted group reviews.

\b(?:I|we)(?:’m| am| are)? (?:hold|own|bought|sold|added|trimmed|   
exited|cut|initiated|opened|closed|long|short)\b   
|\bmy (?:position|book|portfolio|holdings|stake|entry|exit|stop)\b

<table><tr><td>Setting</td><td>Value</td><td>Interpretation</td></tr><tr><td>Recent board posts</td><td>Up to 20</td><td>Selected from published history available at the cycle boundary.</td></tr><tr><td>Post length</td><td>2,000 characters</td><td>Maximum size of a submitted board contribution.</td></tr><tr><td>Board-view text budget</td><td>48,000 characters</td><td>An aggregate bound that can further limit presented material.</td></tr><tr><td>Memory publication</td><td>Up to 6,000 characters</td><td>A prefix of the configured strategy-memory file, not a generated summary.</td></tr><tr><td>Memory-view text budget</td><td>60,000 characters</td><td>Latest available peer publications are subject to an aggregate bound.</td></tr><tr><td>Structured pings</td><td>Up to 3 per cycle</td><td>Applies to the board with pings.</td></tr><tr><td>Incoming mention wakes</td><td>Up to 12 per recipient per UTC day</td><td>Bounds additional activations rather than ordinary reading or posting.</td></tr><tr><td>Recipient cooldown</td><td>5 minutes</td><td>Further wakes within the cooldown are suppressed. Published messages</td></tr><tr><td>Maximum causal depth</td><td>4</td><td>remain visible. Bounds a chain of mention-triggered activations, not lifetime participation.</td></tr></table>

Table 10: Channel settings on 8 October. The limits concern communication between managers, not research.

## J.4 Peer memory

You are a member of a trading swarm with shared memory snapshots, with each member managing a separate account. Your objective remains the SPY-relative alpha defined in your decision contract. Your swarm provides an individual-return scorecard and readPeerMemory, which gives readonly access to bounded, published snapshots of other members’ configured memory files. These snapshots come from successfully completed cycles and are fixed for your current cycle; they are not live access to another agent’s working files. A bounded snapshot of your own configured memory file is also made available to your swarm after a successful cycle when publication succeeds. This environment provides no message board or agent pings. You retain trading authority only over your own account.

## K Post-census patterns

The census applies case-insensitive regular expressions to the body of each published post. A post counts once per measure however many matches it contains. “Names a peer” requires a pseudonym of the form Agent NN other than the author’s own. The other measures use the following patterns.

## Own positions:

## Verification request:

|\bverify before (?:use|using|you|acting|relying|trading)\b

|\bcheck (?:them|this|it) before you (?:trade|act|rely)\b

## Not an instruction or copy:

not (?:an? )?(?:trade |trading )?(?:instruction|recommendation|

directive|signal to (?:copy|follow))

|not a (?:clone|copy)\b|not (?:copying|cloning)\b

## Claimed independent verification:

\bindependently (?:verified|checked|confirmed|read|recovered|

repeated|retrieved|rechecked)\b

|\bindependent (?:audit|check|verification|recovery|confirmation)\b

Long tokens are whitespace-delimited tokens longer than 20 characters, as a share of all tokens. Convention adoption uses one pattern per group:

– the verification-request pattern for Opus’s collaborative board;

– \bverify independently\b for Opus’s board with pings;

– not (?:an? )?(?:trade |trading )?instruction for Grok’s board with pings.

A peer use counts as delivered when it appears inside a peer-generated content block of the view recorded for the adopter’s cycle.

The by-day comparison of position disclosure in Section 6.2 uses a broader pattern than the census: a first-person subject (I or we, with contractions and an optional adverb) followed by a verb of holding or trading (hold, own, bought, sold, trimmed, added, exited, entered, cut, reduced, covered and their other forms); my or our followed within two words by position, book, portfolio, stake, shares, holdings, entry, exit, stop, long or weight; or a label such as “Positions:”, “Holdings:”, “Book:” or “Portfolio:”. Rendered views are HTML-escaped and are unescaped before any matching.